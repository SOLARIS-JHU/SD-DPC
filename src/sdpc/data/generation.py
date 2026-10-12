"""Dataset generation for system identification and policy training."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from neuromancer.dataset import DictDataset
from torch.utils.data import DataLoader

__all__ = [
    "get_data",
    "get_box_policy_data",
    "get_obstacle_policy_data",
    "get_data_discrete",
]


def _make_loaders(train_dict, dev_dict, batch_size, *, shuffle):
    train_data = DictDataset(train_dict, name="train")
    dev_data = DictDataset(dev_dict, name="dev")
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=train_data.collate_fn,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dev_data.collate_fn,
    )
    return train_loader, dev_loader


def get_data(
    sys,
    nsim: int,
    nsteps: int,
    ts: float,
    bs: int,
    device: torch.device = torch.device("cpu"),
    measurement_noise: float = 0.0,
) -> Tuple[DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    """Generate system-ID trajectories from a PSL system.

    ``measurement_noise`` is the standard deviation of additive Gaussian state
    measurement noise, expressed as a fraction of each split's per-state empirical
    standard deviation. Inputs are never perturbed. A value of ``0.1`` therefore
    gives 10% state measurement noise while preserving the clean simulated controls.
    """
    if measurement_noise < 0.0:
        raise ValueError("measurement_noise must be non-negative")
    nx, nu = sys.nx, sys.nu
    nbatch = nsim // nsteps
    if nbatch < 1:
        raise ValueError("nsim must be at least nsteps")
    length = nbatch * nsteps
    train_sim, dev_sim, test_sim = [sys.simulate(nsim=nsim, ts=ts) for _ in range(3)]

    def _tensorise(sim):
        X = torch.as_tensor(sim["X"][:length], dtype=torch.float32, device=device)
        U = torch.as_tensor(sim["U"][:length], dtype=torch.float32, device=device)
        if measurement_noise:
            scale = X.std(dim=0, unbiased=False).clamp_min(torch.finfo(X.dtype).eps)
            X = X + measurement_noise * scale * torch.randn_like(X)
        X = X.reshape(nbatch, nsteps, nx)
        U = U.reshape(nbatch, nsteps, nu)

        out = {"X": X, "xn": X[:, 0:1, :], "u": U}
        for i in range(nx):
            xi = X[:, :, i:i + 1]
            out[f"x{i}"] = xi
            out[f"x{i}_n"] = xi[:, 0:1, :]
        return out

    train_dict = _tensorise(train_sim)
    dev_dict = _tensorise(dev_sim)
    train_loader, dev_loader = _make_loaders(train_dict, dev_dict, bs, shuffle=True)
    return train_loader, dev_loader, _tensorise(test_sim)


def get_box_policy_data(
    nsteps: int,
    n_samples: int,
    nx: int,
    device: torch.device,
    *,
    xmin: float = -1.0,
    xmax: float = 1.0,
    same_ref_for_all_states: bool = True,
    zero_refs: bool = False,
    batch_size: int = 200,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Generate box-sampled initial states and constant policy references."""
    gen_train = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    gen_dev = None if seed is None else torch.Generator(device=device).manual_seed(seed + 1)

    def _build_split(gen):
        ref_width = 1 if same_ref_for_all_states else nx
        if zero_refs:
            levels = torch.zeros(n_samples, 1, ref_width, device=device)
        else:
            levels = torch.rand(n_samples, 1, ref_width, device=device, generator=gen)
        if same_ref_for_all_states:
            levels = levels.repeat(1, 1, nx)
        ref = levels.repeat(1, nsteps + 1, 1)

        xn = torch.rand(n_samples, 1, nx, device=device, generator=gen)
        xn = xn * (xmax - xmin) + xmin
        data = {"xn": xn, "r": ref}
        for i in range(nx):
            data[f"x{i}_n"] = xn[:, :, i:i + 1]
            data[f"r{i}"] = ref[:, :, i:i + 1]
        return data

    train_dict = _build_split(gen_train)
    dev_dict = _build_split(gen_dev)
    return _make_loaders(train_dict, dev_dict, batch_size, shuffle=False)


def get_obstacle_policy_data(
    nsteps: int,
    n_samples: int,
    nx: int,
    device: torch.device,
    *,
    pos_idx: Tuple[int, int] = (0, 2),
    batch_size: int = 200,
    seed: Optional[int] = None,
    init_rect: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, -1.0), (1.0, 1.0)),
    ref_rect: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, -1.0), (1.0, 1.0)),
    ellipse_configs: Optional[Dict[str, float]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Generate initial positions and references outside a keep-out ellipse."""
    ellipse_configs = ellipse_configs or {"p": 10.0, "b": 8.0, "c": 1.0, "d": 2.0}
    p = float(ellipse_configs["p"])
    b = float(ellipse_configs["b"])
    c = float(ellipse_configs["c"])
    d = float(ellipse_configs["d"])
    if b <= 0.0 or p < 0.0:
        raise ValueError("ellipse parameters require b > 0 and p >= 0")
    if max(pos_idx) >= nx or min(pos_idx) < 0 or pos_idx[0] == pos_idx[1]:
        raise ValueError("pos_idx must select two distinct state dimensions")

    gen_train = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    gen_dev = None if seed is None else torch.Generator(device=device).manual_seed(seed + 1)

    def _sample_outside(n, rect, gen, max_iters=100, chunk=4096):
        (x0, y0), (x1, y1) = rect
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("sampling rectangles must have positive width and height")

        blocks = []
        count = 0
        for _ in range(max_iters):
            if count >= n:
                break
            size = max(n - count, chunk)
            xs = x0 + (x1 - x0) * torch.rand(size, device=device, generator=gen)
            ys = y0 + (y1 - y0) * torch.rand(size, device=device, generator=gen)
            keep = b * (xs - c) ** 2 + (ys - d) ** 2 >= (0.5 * p) ** 2
            if keep.any():
                block = torch.stack((xs[keep], ys[keep]), dim=1)
                blocks.append(block)
                count += block.shape[0]
        if count < n:
            raise RuntimeError(
                f"could not sample {n} points outside the ellipse within the rectangle"
            )
        return torch.cat(blocks, dim=0)[:n]

    def _build_split(gen):
        xy0 = _sample_outside(n_samples, init_rect, gen)
        xn = torch.zeros(n_samples, 1, nx, device=device)
        xn[:, 0, pos_idx[0]] = xy0[:, 0]
        xn[:, 0, pos_idx[1]] = xy0[:, 1]

        xy_ref = _sample_outside(n_samples, ref_rect, gen)
        ref = xy_ref[:, None, :].repeat(1, nsteps + 1, 1)
        return {
            "xn": xn,
            "r": ref,
            "obs_p": torch.full((n_samples, 1, 1), p, device=device),
            "obs_b": torch.full((n_samples, 1, 1), b, device=device),
            "obs_c": torch.full((n_samples, 1, 1), c, device=device),
            "obs_d": torch.full((n_samples, 1, 1), d, device=device),
        }

    train_dict = _build_split(gen_train)
    dev_dict = _build_split(gen_dev)
    return _make_loaders(train_dict, dev_dict, batch_size, shuffle=False)


def get_data_discrete(
    sys,
    nsim: int,
    nsteps: int,
    ts: float,
    bs: int,
    device: torch.device = torch.device("cpu"),
    x0_range: Tuple[float, float] = (-20.0, 20.0),
) -> Tuple[DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    """System-ID data for a discrete plant exposing ``nx, nu, step(x,u), get_U, sample_x0``.

    Produces ``nbatch = nsim // nsteps`` independent length-``nsteps`` sequences per split,
    with the same dict keys/shapes as :func:`get_data` (``X, xn, u, x{i}, x{i}_n``).
    """
    nx, nu = sys.nx, sys.nu
    nbatch = nsim // nsteps
    length = nbatch * nsteps

    def _make_split():
        X = torch.empty(nbatch, nsteps, nx, dtype=torch.float32, device=device)
        U = torch.empty(nbatch, nsteps, nu, dtype=torch.float32, device=device)
        low, high = float(x0_range[0]), float(x0_range[1])
        for b in range(nbatch):
            U_seq = sys.get_U(nsteps).to(device=device, dtype=torch.float32)
            U[b] = U_seq
            x = sys.sample_x0(low=low, high=high).to(device=device, dtype=torch.float32)
            for k in range(nsteps):
                X[b, k, :] = x
                x = sys.step(x, U_seq[k, :])
        return {"X": X, "U": U}

    def _tensorise(sim: Dict[str, torch.Tensor], keep_1d: bool = True) -> Dict[str, torch.Tensor]:
        X_t, U_t = sim["X"], sim["U"]
        out = {"X": X_t, "xn": X_t[:, 0:1, :], "u": U_t}
        if keep_1d:
            for i in range(nx):
                xi = X_t[:, :, i:i + 1]
                out[f"x{i}"] = xi
                out[f"x{i}_n"] = xi[:, 0:1, :]
        return out

    train_dict = _tensorise(_make_split())
    dev_dict = _tensorise(_make_split())

    train_data = DictDataset(train_dict, name="train")
    dev_data = DictDataset(dev_dict, name="dev")
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=bs, shuffle=True, collate_fn=dev_data.collate_fn)

    # single long test rollout
    low, high = float(x0_range[0]), float(x0_range[1])
    U_test = sys.get_U(length).to(device=device, dtype=torch.float32)
    X_test = torch.empty(length, nx, dtype=torch.float32, device=device)
    x = sys.sample_x0(low=low, high=high).to(device=device, dtype=torch.float32)
    for k in range(length):
        X_test[k, :] = x
        x = sys.step(x, U_test[k, :])
    test_dict = _tensorise({"X": X_test.reshape(1, length, nx), "U": U_test.reshape(1, length, nu)})
    return train_loader, dev_loader, test_dict
