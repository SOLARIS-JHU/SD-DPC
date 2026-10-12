"""Run-directory management, config/seed snapshotting, and lightweight logging.

A "run" is one execution of a stage (system ID, policy training, evaluation).
Each run gets a directory under the example's ``results/`` tree; the exact config and
seed are snapshotted next to the outputs so any result is reproducible.

Nothing here ever deletes or overwrites existing content: :func:`new_run_dir`
auto-increments ``run_1, run_2, ...`` and refuses to touch a directory that exists.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml

__all__ = ["new_run_dir", "snapshot_config", "save_json", "save_yaml", "CustomLogger"]


class CustomLogger:
    """Small stdout/artifact logger compatible with ``SparseTrainer``."""

    def __init__(
        self,
        args=None,
        savedir="test",
        verbosity=10,
        stdout=(
            "nstep_dev_loss",
            "loop_dev_loss",
            "best_loop_dev_loss",
            "nstep_dev_ref_loss",
            "loop_dev_ref_loss",
        ),
    ):
        self.savedir = Path(savedir)
        self.savedir.mkdir(parents=True, exist_ok=True)
        self.stdout = tuple(stdout)
        self.verbosity = int(verbosity)
        self.start_time = time.time()
        self.step = 0
        self.args = args
        print(args)

    def log_weights(self, model):
        nweights = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {nweights}")
        return nweights

    def log_metrics(self, output, step=None):
        self.step = self.step if step is None else int(step)
        if self.verbosity <= 0 or self.step % self.verbosity:
            return
        entries = [f"epoch: {self.step}"]
        for key, value in output.items():
            if key not in self.stdout:
                continue
            try:
                entries.append(f"{key}: {value.item():.3e}")
            except (AttributeError, ValueError):
                continue
        entries.append(f"eltime: {time.time() - self.start_time: .5f}")
        print("\t".join(entry for entry in entries if "reg_error" not in entry))

    def log_artifacts(self, artifacts):
        for name, artifact in artifacts.items():
            payload = artifact.state_dict() if isinstance(artifact, torch.nn.Module) else artifact
            torch.save(payload, self.savedir / name)

    def clean_up(self):
        pass


def new_run_dir(base_dir: Union[str, Path], prefix: str = "run") -> Path:
    """Create and return a fresh, auto-incremented ``<base_dir>/<prefix>_<n>`` directory."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while (base / f"{prefix}_{n}").exists():
        n += 1
    run_dir = base / f"{prefix}_{n}"
    run_dir.mkdir(parents=True)
    return run_dir


def snapshot_config(run_dir: Union[str, Path], cfg: Dict[str, Any],
                    seed: Optional[int] = None) -> None:
    """Write the resolved config (and seed) into ``run_dir`` for reproducibility."""
    run_dir = Path(run_dir)
    meta = {k: v for k, v in cfg.items() if not k.startswith("_")}
    meta["_snapshot_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if seed is not None:
        meta["seed"] = seed
    save_yaml(run_dir / "config.yaml", meta)


def save_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def save_yaml(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False, default_flow_style=False)
