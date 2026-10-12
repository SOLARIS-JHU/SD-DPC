"""Uniform checkpoint layout and discovery across all case studies."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from .run import new_run_dir

__all__ = [
    "new_policy_run_dir",
    "find_dynamics_checkpoint",
    "find_policy_checkpoint",
    "find_nn_checkpoint",
]


def _run_number(path: Path) -> int:
    suffix = path.name.removeprefix("run_")
    return int(suffix) if suffix.isdigit() else -1


def _latest_run(base: Path) -> Optional[Path]:
    runs = sorted(base.glob("run_*"), key=_run_number)
    return runs[-1] if runs else None


def _latest_checkpoint(base: Path, relative: Path) -> Optional[Path]:
    """Return the newest numerically ordered run containing ``relative``."""
    completed_runs = [run for run in base.glob("run_*") if (run / relative).exists()]
    if not completed_runs:
        return None
    return max(completed_runs, key=_run_number) / relative


def _explicit_path(results_dir: Path, value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (results_dir.parent / path).resolve()


def new_policy_run_dir(results_dir: Union[str, Path], policy_type: str) -> Path:
    """Create ``models/policies/{sd_dpc|nn_dpc}/run_N``."""
    policy_type = str(policy_type).lower()
    branch = {"sparse": "sd_dpc", "sd_dpc": "sd_dpc", "nn": "nn_dpc", "nn_dpc": "nn_dpc"}
    if policy_type not in branch:
        raise ValueError("policy_type must be sparse/sd_dpc or nn/nn_dpc")
    return new_run_dir(Path(results_dir) / "models" / "policies" / branch[policy_type])


def find_dynamics_checkpoint(results_dir: Union[str, Path], cfg: Optional[Dict] = None) -> Path:
    results_dir, cfg = Path(results_dir), cfg or {}
    if cfg.get("sindy_path"):
        return _explicit_path(results_dir, cfg["sindy_path"])
    path = _latest_checkpoint(
        results_dir / "models" / "dynamics", Path("saved_models") / "sindy.pt"
    )
    if path is None:
        raise FileNotFoundError(f"no dynamics runs under {results_dir / 'models' / 'dynamics'}")
    return path


def find_policy_checkpoint(results_dir: Union[str, Path], cfg: Optional[Dict] = None) -> Path:
    """Find the latest SD-DPC policy, preferring the uniform layout."""
    results_dir, cfg = Path(results_dir), cfg or {}
    if cfg.get("policy_path"):
        return _explicit_path(results_dir, cfg["policy_path"])
    path = _latest_checkpoint(
        results_dir / "models" / "policies" / "sd_dpc",
        Path("saved_models") / "policy_sparse.pt",
    )
    if path is not None:
        return path
    legacy = _latest_checkpoint(
        results_dir / "models", Path("SparseDPC") / "saved_models" / "policy_sparse.pt"
    )
    if legacy is not None:
        return legacy
    raise FileNotFoundError(f"no SD-DPC checkpoints under {results_dir / 'models'}")


def find_nn_checkpoint(results_dir: Union[str, Path], cfg: Optional[Dict] = None) -> Path:
    """Find the latest NN-DPC policy, preferring the uniform layout."""
    results_dir, cfg = Path(results_dir), cfg or {}
    if cfg.get("nn_policy_path"):
        return _explicit_path(results_dir, cfg["nn_policy_path"])
    path = _latest_checkpoint(
        results_dir / "models" / "policies" / "nn_dpc",
        Path("saved_models") / "policy_nn.pth",
    )
    if path is not None:
        return path
    legacy = _latest_checkpoint(
        results_dir / "models", Path("NN-DPC") / "saved_models" / "policy_nn.pth"
    )
    if legacy is not None:
        return legacy
    static_legacy = (
        results_dir / "models" / "neural" / "NN-DPC" / "saved_models" / "policy_nn.pth"
    )
    if static_legacy.exists():
        return static_legacy
    raise FileNotFoundError(f"no NN-DPC checkpoints under {results_dir / 'models'}")
