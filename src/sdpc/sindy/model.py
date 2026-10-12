import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict
import torch
import torch.nn as nn
from neuromancer.dynamics.ode import ODESystem

from .library import LibraryPlan, CompiledFunctionLibrary


class SINDyVectorized(ODESystem):
    """
    Multi-output SINDy with per-state coefficient vectors (variable length).
    Uses a CompiledFunctionLibrary and recompiles a fast plan for the
    union of active columns across states (so forward computes only needed Θ cols).
    """
    def __init__(self,
                 library: CompiledFunctionLibrary,
                 n_out: Optional[int] = None,
                 policy_name: Optional[list] = None,
                 seed: Optional[int] = None,
                 device: torch.device = torch.device("cpu")):
        assert isinstance(library, CompiledFunctionLibrary), "`library` invalid"
        self.library = library
        self.n_out = n_out or library.n_features
        self.policy_name = policy_name

        super().__init__(library.shape[1], self.n_out)

        n_terms = library.shape[0]
        gen = None if seed is None else torch.Generator(device=device).manual_seed(seed)

        # per-state active global columns + coef vectors
        self.active_idx: List[List[int]] = [list(range(n_terms)) for _ in range(self.n_out)]
        self.Xi = nn.ParameterList([
            nn.Parameter(torch.empty(n_terms, 1, device=device).uniform_(-0.05, 0.05, generator=gen))
            for _ in range(self.n_out)
        ])

        self.function_names = list(library.function_names)
        self._plan: LibraryPlan = self._recompile_fast_path(device=device)

        self.float()

    # compile fast plan for the union of active terms (called at init and after pruning)
    def _recompile_fast_path(self, device=None) -> LibraryPlan:
        union = sorted(set(idx for s in self.active_idx for idx in s))
        plan = self.library.compile(union, device=device)
        # Build per-state mapping from global -> local-in-plan (LongTensor)
        self._state_local_idx: List[torch.Tensor] = []
        for s in self.active_idx:
            g = torch.tensor(s, dtype=torch.long, device=plan.global2local.device)
            l = plan.global2local[g]
            assert (l >= 0).all(), "Plan missing some active columns."
            self._state_local_idx.append(l)
        return plan

    # read-only dense view
    @property
    def coef(self) -> torch.Tensor:
        device = self.Xi[0].device
        n_terms = len(self.function_names)
        C = torch.zeros(n_terms, self.n_out, device=device)
        for i in range(self.n_out):
            idx = torch.tensor(self.active_idx[i], dtype=torch.long, device=device)
            C[idx, i] = self.Xi[i].squeeze(1)
        return C

    def ode_equations(self, x: torch.Tensor, u: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.to(self.Xi[0].device)
        if u is not None: u = u.to(self.Xi[0].device)
        Theta_sub = self.library.evaluate_plan(x, u, self._plan)  # (B, M_union)

        rhs = []
        for i in range(self.n_out):
            Theta_i = Theta_sub.index_select(1, self._state_local_idx[i])  # (B, n_i)
            rhs_i = Theta_i @ self.Xi[i]                                   # (B,1)
            rhs.append(rhs_i)
        return torch.cat(rhs, dim=1)                                       # (B, n_out)

    @torch.no_grad()
    def prune_(self,
               thresholds: float | Sequence[float] = 1e-3,
               *,
               noise_after: float = 0.0,
               nonzero_only: bool = True,
               eps: float = 0.0) -> Dict:
        """Hard magnitude pruning per state; shrinks Xi[i] and active_idx[i], then recompiles plan."""
        if isinstance(thresholds, (int, float)):
            thr = [float(thresholds)] * self.n_out
        else:
            thr = list(thresholds); assert len(thr) == self.n_out

        changed_states = []
        kept_counts, removed_counts = [], []

        for i in range(self.n_out):
            coef_i = self.Xi[i].squeeze(1)                         # (n_i,)
            keep_mask = (coef_i.abs() >= thr[i])
            if keep_mask.sum().item() == 0:
                keep_mask[torch.argmax(coef_i.abs())] = True

            if keep_mask.sum().item() < coef_i.numel():
                new_loc = torch.nonzero(keep_mask, as_tuple=False).squeeze(1).tolist()
                self.active_idx[i] = [self.active_idx[i][k] for k in new_loc]

                kept_counts.append(len(self.active_idx[i]))
                removed_counts.append(int(coef_i.numel() - len(self.active_idx[i])))

                new_coef = self.Xi[i][keep_mask, :].detach().clone()
                if noise_after and noise_after > 0.0:
                    if nonzero_only:
                        nz = (new_coef.abs() > eps)
                        new_coef[nz] += noise_after * torch.randn_like(new_coef[nz])
                    else:
                        new_coef += noise_after * torch.randn_like(new_coef)

                self.Xi[i] = nn.Parameter(new_coef, requires_grad=True)
                changed_states.append(i)
            else:
                kept_counts.append(int(coef_i.numel()))
                removed_counts.append(0)

        # recompile fast plan for the union; refresh per-state local indices
        self._plan = self._recompile_fast_path(device=self.Xi[0].device)

        return {
            "changed": len(changed_states) > 0,
            "changed_states": changed_states,
            "kept_counts": kept_counts,
            "removed_counts": removed_counts,
        }

    @torch.no_grad()
    def pretty_print(self, labels: Optional[List[str]] = None, coef_tol: float = 1e-12):
        if labels is None:
            labels = [f"x{i}" for i in range(self.n_out)]
        for i in range(self.n_out):
            names_i = [self.function_names[j] for j in self.active_idx[i]]
            coefs_i = self.Xi[i].squeeze(1).detach().cpu().tolist()
            terms = [f"{c:+.4e}·{n}" for n, c in zip(names_i, coefs_i) if abs(c) > coef_tol]
            rhs = " + ".join(terms) if terms else "0"
            lhs = self.policy_name[i] if self.policy_name else f"d{labels[i]}/dt"
            print(f"{lhs} = {rhs}")

    @torch.no_grad()
    def add_terms_(self, state: int, global_indices: Sequence[int], init_scale: float = 1e-3):
        device = self.Xi[state].device
        cur = set(self.active_idx[state])
        to_add = [g for g in global_indices if g not in cur]
        if not to_add: return
        self.active_idx[state].extend(to_add)
        add_coef = init_scale * torch.randn(len(to_add), 1, device=device)
        self.Xi[state] = nn.Parameter(torch.cat([self.Xi[state].data, add_coef], dim=0), requires_grad=True)
        self._plan = self._recompile_fast_path(device=device)
