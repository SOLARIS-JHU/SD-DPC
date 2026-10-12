import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict
import torch
import torch.nn as nn
from neuromancer.dynamics.ode import ODESystem
from typing import Optional, Sequence, List, Dict


@dataclass
class LibraryPlan:
    subset_global: torch.Tensor
    order: torch.Tensor
    take_bias: bool
    x_idx: Optional[torch.Tensor]
    u_idx: Optional[torch.Tensor]
    sqrt_x_idx: Optional[torch.Tensor]
    poly_sel: Dict[int, torch.Tensor]
    uu_pairs_sel: Optional[torch.Tensor]
    xu_sel: Optional[torch.Tensor]
    fourier_sin_sel: Optional[torch.Tensor]
    fourier_cos_sel: Optional[torch.Tensor]
    # NEW: Fourier on reference (policy) inputs
    fourier_sin_r_sel: Optional[torch.Tensor]
    fourier_cos_r_sel: Optional[torch.Tensor]
    global2local: torch.Tensor


class CompiledFunctionLibrary:
    """
    Θ(X,U) with a compiled layout. If `is_policy=True`, control-like inputs
    are *named* 'r' (reference) instead of 'u' in `function_names`.
    Computation is unchanged: the second argument is still `U` (shape (B, n_control)).
    """
    def __init__(self,
                 n_features: int,
                 n_control: int,
                 *,
                 include_bias: bool = True,
                 max_degree: int = 2,
                 add_sqrt: bool = False,
                 add_xu: bool = True,
                 add_u_prod: bool = True,
                 add_fourier: bool = False,
                 max_freq: int = 1,
                 is_policy: bool = False):
        self.n_features = n_features
        self.n_control = n_control
        self.include_bias = include_bias
        self.max_degree = max_degree
        self.add_sqrt = add_sqrt
        self.add_xu = add_xu
        self.add_u_prod = add_u_prod
        self.add_fourier = add_fourier
        self.max_freq = max_freq
        self.is_policy = is_policy

        # ----- build static layout (offsets) once -----
        self.offsets: Dict[str, Tuple[int, int]] = {}
        self._poly_combos: Dict[int, torch.Tensor] = {}
        self._poly_deg_offsets: Dict[int, Tuple[int,int]] = {}
        self._uu_pairs = None
        self._freqs = None

        start = 0
        if include_bias:
            self.offsets["bias"] = (start, start+1); start += 1
        self.offsets["x"] = (start, start + n_features); start += n_features
        if n_control > 0:
            self.offsets["u"] = (start, start + n_control); start += n_control
        if add_sqrt:
            self.offsets["sqrt_x"] = (start, start + n_features); start += n_features

        if max_degree >= 2:
            for d in range(2, max_degree+1):
                combos = list(itertools.combinations_with_replacement(range(n_features), d))
                if combos:
                    self._poly_combos[d] = torch.tensor(combos, dtype=torch.long)
                    m_d = len(combos)
                    self._poly_deg_offsets[d] = (start, start + m_d); start += m_d

        if add_u_prod and n_control >= 2:
            p_idx, q_idx = torch.triu_indices(n_control, n_control, offset=1)
            self._uu_pairs = torch.stack([p_idx, q_idx], dim=0)  # (2, n_pairs)
            self.offsets["uu"] = (start, start + self._uu_pairs.shape[1]); start += self._uu_pairs.shape[1]

        if add_xu and n_control > 0:
            self.offsets["xu"] = (start, start + n_features*n_control); start += n_features*n_control

        if add_fourier and max_freq >= 1:
            self._freqs = torch.arange(1, max_freq+1).long()
            # Fourier on states x
            self.offsets["fourier_sin"] = (start, start + n_features*max_freq); start += n_features*max_freq
            self.offsets["fourier_cos"] = (start, start + n_features*max_freq); start += n_features*max_freq
            # NEW: Fourier on reference inputs r (only when used as policy inputs)
            if is_policy and n_control > 0:
                self.offsets["fourier_sin_r"] = (start, start + n_control*max_freq); start += n_control*max_freq
                self.offsets["fourier_cos_r"] = (start, start + n_control*max_freq); start += n_control*max_freq

        # Total shape (n_terms, nx+nu)
        self.shape = (start, n_features + n_control)

        # Build names once (using 'u' or 'r' according to is_policy)
        self.function_names: List[str] = self._build_names()

    # ---------- helper to (re)build names ----------
    def _build_names(self) -> List[str]:
        ctrl = "r" if self.is_policy else "u"
        names: List[str] = []

        # Respect existing offsets; only generate textual names
        if "bias" in self.offsets:
            names.append("1")

        # linear x
        nx = self.n_features
        names.extend([f"x{i}" for i in range(nx)])

        # linear controls
        if "u" in self.offsets:
            nu = self.n_control
            names.extend([f"{ctrl}{j}" for j in range(nu)])  # label is 'r' if policy

        # sqrt(x)
        if "sqrt_x" in self.offsets:
            names.extend([f"sqrt(x{i})" for i in range(nx)])

        # poly(x)
        for d in sorted(self._poly_combos.keys()):
            combos = self._poly_combos[d].tolist()
            for combo in combos:
                counts = {}
                for idx in combo: counts[idx] = counts.get(idx, 0) + 1
                names.append("*".join([f"x{k}" if p==1 else f"x{k}^{p}" for k,p in sorted(counts.items())]))

        # uu
        if self._uu_pairs is not None:
            for p, q in zip(self._uu_pairs[0].tolist(), self._uu_pairs[1].tolist()):
                names.append(f"{ctrl}{p}*{ctrl}{q}")

        # xu
        if "xu" in self.offsets:
            for i in range(nx):
                for j in range(self.n_control):
                    names.append(f"x{i}*{ctrl}{j}")

        # fourier on x
        if "fourier_sin" in self.offsets:
            K = (self.offsets["fourier_sin"][1] - self.offsets["fourier_sin"][0]) // nx
            for i in range(nx):
                for k in range(1, K+1):
                    names.append(f"sin({k}·x{i})")
        if "fourier_cos" in self.offsets:
            K = (self.offsets["fourier_cos"][1] - self.offsets["fourier_cos"][0]) // nx
            for i in range(nx):
                for k in range(1, K+1):
                    names.append(f"cos({k}·x{i})")

        # NEW: fourier on reference inputs r (policy naming)
        if "fourier_sin_r" in self.offsets:
            nu = self.n_control
            K = (self.offsets["fourier_sin_r"][1] - self.offsets["fourier_sin_r"][0]) // nu
            for j in range(nu):
                for k in range(1, K+1):
                    names.append(f"sin({k}·{ctrl}{j})")
        if "fourier_cos_r" in self.offsets:
            nu = self.n_control
            K = (self.offsets["fourier_cos_r"][1] - self.offsets["fourier_cos_r"][0]) // nu
            for j in range(nu):
                for k in range(1, K+1):
                    names.append(f"cos({k}·{ctrl}{j})")

        assert len(names) == self.shape[0]
        return names

    # Public toggle if you ever want to switch after creation
    def set_policy_names(self, is_policy: bool = True):
        self.is_policy = bool(is_policy)
        self.function_names = self._build_names()

    # ---- compile & evaluate ----
    def compile(self, subset_global: Optional[Sequence[int]] = None, device: Optional[torch.device] = None) -> LibraryPlan:
        n_terms = self.shape[0]
        if subset_global is None:
            subset = torch.arange(n_terms, dtype=torch.long, device=device)
        else:
            subset = torch.as_tensor(subset_global, dtype=torch.long, device=device)
        subset, _ = torch.sort(subset)

        def in_range(r):
            s,e = r
            mask = (subset >= s) & (subset < e)
            idx_local = subset[mask] - s
            return mask, idx_local

        take_bias = False
        x_idx = u_idx = sqrt_x_idx = None
        poly_sel: Dict[int, torch.Tensor] = {}
        uu_pairs_sel = xu_sel = fourier_sin_sel = fourier_cos_sel = None
        # NEW:
        fourier_sin_r_sel = fourier_cos_r_sel = None

        if "bias" in self.offsets:
            mask_bias, _ = in_range(self.offsets["bias"])
            take_bias = bool(mask_bias.any().item())

        mask_x, x_idx = in_range(self.offsets["x"])
        if not mask_x.any(): x_idx = None

        if "u" in self.offsets:
            mask_u, u_idx = in_range(self.offsets["u"])
            if not mask_u.any(): u_idx = None

        if "sqrt_x" in self.offsets:
            mask_sx, sqrt_x_idx = in_range(self.offsets["sqrt_x"])
            if not mask_sx.any(): sqrt_x_idx = None

        for d, combos in self._poly_combos.items():
            s,e = self._poly_deg_offsets[d]
            mask_pd, idx_pd = in_range((s,e))
            if mask_pd.any():
                poly_sel[d] = idx_pd

        if "uu" in self.offsets and self._uu_pairs is not None:
            s,e = self.offsets["uu"]
            mask_uu, idx_uu = in_range((s,e))
            if mask_uu.any():
                uu_pairs_sel = self._uu_pairs[:, idx_uu]

        if "xu" in self.offsets:
            s,e = self.offsets["xu"]
            mask_xu, idx_xu = in_range((s,e))
            if mask_xu.any():
                nx, nu = self.n_features, self.n_control
                i_idx = idx_xu // nu
                j_idx = idx_xu %  nu
                xu_sel = torch.stack([i_idx, j_idx], dim=0)

        if "fourier_sin" in self.offsets:
            s,e = self.offsets["fourier_sin"]
            mask_fs, idx_fs = in_range((s,e))
            if mask_fs.any():
                nx, K = self.n_features, (e - s) // self.n_features
                i_idx = idx_fs // K
                k_idx0 = idx_fs %  K
                fourier_sin_sel = torch.stack([i_idx, k_idx0], dim=0)

        if "fourier_cos" in self.offsets:
            s,e = self.offsets["fourier_cos"]
            mask_fc, idx_fc = in_range((s,e))
            if mask_fc.any():
                nx, K = self.n_features, (e - s) // self.n_features
                i_idx = idx_fc // K
                k_idx0 = idx_fc %  K
                fourier_cos_sel = torch.stack([i_idx, k_idx0], dim=0)

        # NEW: reference Fourier selectors
        if "fourier_sin_r" in self.offsets:
            s,e = self.offsets["fourier_sin_r"]
            mask_fsr, idx_fsr = in_range((s,e))
            if mask_fsr.any():
                nu, K = self.n_control, (e - s) // self.n_control
                j_idx = idx_fsr // K
                k_idx0 = idx_fsr %  K
                fourier_sin_r_sel = torch.stack([j_idx, k_idx0], dim=0)

        if "fourier_cos_r" in self.offsets:
            s,e = self.offsets["fourier_cos_r"]
            mask_fcr, idx_fcr = in_range((s,e))
            if mask_fcr.any():
                nu, K = self.n_control, (e - s) // self.n_control
                j_idx = idx_fcr // K
                k_idx0 = idx_fcr %  K
                fourier_cos_r_sel = torch.stack([j_idx, k_idx0], dim=0)

        g2l = torch.full((self.shape[0],), -1, dtype=torch.long, device=subset.device)
        g2l[subset] = torch.arange(subset.numel(), device=subset.device)

        return LibraryPlan(
            subset_global=subset,
            order=subset.clone(),
            take_bias=take_bias,
            x_idx=x_idx,
            u_idx=u_idx,
            sqrt_x_idx=sqrt_x_idx,
            poly_sel=poly_sel,
            uu_pairs_sel=uu_pairs_sel,
            xu_sel=xu_sel,
            fourier_sin_sel=fourier_sin_sel,
            fourier_cos_sel=fourier_cos_sel,
            fourier_sin_r_sel=fourier_sin_r_sel,
            fourier_cos_r_sel=fourier_cos_r_sel,
            global2local=g2l
        )

    def evaluate_plan(self, x: torch.Tensor, u: Optional[torch.Tensor], plan: LibraryPlan) -> torch.Tensor:
        B = x.shape[0]; device = x.device
        outs = []
        if plan.take_bias: outs.append(torch.ones(B, 1, device=device))
        if plan.x_idx is not None: outs.append(x[:, plan.x_idx])
        if plan.u_idx is not None: outs.append(u[:, plan.u_idx])
        if plan.sqrt_x_idx is not None: outs.append(torch.sqrt(torch.clamp(x[:, plan.sqrt_x_idx], min=1e-12)))
        for d, rows in plan.poly_sel.items():
            combos = self._poly_combos[d].to(device)[rows]
            Xsel = x.index_select(1, combos.reshape(-1)).reshape(B, combos.shape[0], d)
            outs.append(Xsel.prod(dim=2))
        if plan.uu_pairs_sel is not None:
            p_idx = plan.uu_pairs_sel[0].to(device); q_idx = plan.uu_pairs_sel[1].to(device)
            outs.append(u[:, p_idx] * u[:, q_idx])
        if plan.xu_sel is not None:
            i_idx = plan.xu_sel[0].to(device); j_idx = plan.xu_sel[1].to(device)
            outs.append(x[:, i_idx] * u[:, j_idx])
        if plan.fourier_sin_sel is not None:
            i_idx = plan.fourier_sin_sel[0].to(device); k_idx0 = plan.fourier_sin_sel[1].to(device)
            freqs = (k_idx0 + 1).to(x.dtype).view(1, -1)
            outs.append(torch.sin(x[:, i_idx] * freqs))
        if plan.fourier_cos_sel is not None:
            i_idx = plan.fourier_cos_sel[0].to(device); k_idx0 = plan.fourier_cos_sel[1].to(device)
            freqs = (k_idx0 + 1).to(x.dtype).view(1, -1)
            outs.append(torch.cos(x[:, i_idx] * freqs))
        # NEW: Fourier on reference inputs
        if plan.fourier_sin_r_sel is not None:
            j_idx = plan.fourier_sin_r_sel[0].to(device); k_idx0 = plan.fourier_sin_r_sel[1].to(device)
            freqs = (k_idx0 + 1).to(x.dtype).view(1, -1)
            outs.append(torch.sin(u[:, j_idx] * freqs))
        if plan.fourier_cos_r_sel is not None:
            j_idx = plan.fourier_cos_r_sel[0].to(device); k_idx0 = plan.fourier_cos_r_sel[1].to(device)
            freqs = (k_idx0 + 1).to(x.dtype).view(1, -1)
            outs.append(torch.cos(u[:, j_idx] * freqs))
        return torch.cat(outs, dim=1) if outs else torch.empty(B, 0, device=device)

    def evaluate(self, x: torch.Tensor, u: Optional[torch.Tensor]) -> torch.Tensor:
        plan = self.compile(None, device=x.device)
        return self.evaluate_plan(x, u, plan)
