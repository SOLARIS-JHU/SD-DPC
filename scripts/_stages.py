"""Shared implementations behind the case-specific command-line scripts.

The public entry points live in ``scripts/<CaseStudy>/``. Each one fixes the system,
stage, and default YAML config; this module only avoids duplicating stage internals.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402

from sdpc.config import deep_update, load_config, seed_everything  # noqa: E402
from sdpc.eval import evaluate, evaluate_twotank_nominal  # noqa: E402
from sdpc.io import (  # noqa: E402
    CustomLogger,
    find_dynamics_checkpoint,
    find_nn_checkpoint,
    find_policy_checkpoint,
    new_policy_run_dir,
    new_run_dir,
    save_json,
    snapshot_config,
)
from sdpc.plotting import format_metrics_table  # noqa: E402
from sdpc.registry import make_system  # noqa: E402
from sdpc.sindy import (  # noqa: E402
    CompiledFunctionLibrary,
    SINDyVectorized,
    load_model,
    save_model,
)
from sdpc.training import (  # noqa: E402
    build_nn_policy,
    distill_nn_policy,
    evaluate_distilled_policy,
    load_nn_policy,
    make_debias_config,
    make_distillation_scenarios,
    save_distilled_policy,
    train_policy,
    train_sysid,
    validate_reference_results,
)


def _context(
    system_name: str,
    case_dir: Path,
    config_name: str,
    description: str,
    *,
    allow_reference_horizon: bool = False,
) -> Tuple[argparse.Namespace, Dict]:
    default_config = case_dir / "configs" / config_name
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"stage YAML (default: {default_config})",
    )
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    parser.add_argument("--seed", type=int, default=None, help="override the YAML seed")
    if allow_reference_horizon:
        parser.add_argument(
            "--reference-horizon", type=int, default=None,
            help="override the held-current-reference tracking horizon",
        )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    cfg = load_config(config_path)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    seed_everything(seed)
    device = torch.device(args.device)
    system = make_system(system_name, device=device)
    return args, {
        "cfg": cfg,
        "seed": seed,
        "device": device,
        "system": system,
        "case_dir": case_dir,
        "results_dir": case_dir / "results",
    }


def _load_sindy(results_dir: Path, cfg: Dict, device):
    return load_model(find_dynamics_checkpoint(results_dir, cfg), device=device)


def _load_policy(results_dir: Path, cfg: Dict, device):
    return load_model(find_policy_checkpoint(results_dir, cfg), device=device)


def run_system_id(system_name: str, case_dir: Path) -> None:
    _, ctx = _context(system_name, case_dir, "sysid.yaml", "SINDy system identification")
    cfg, system, device = ctx["cfg"], ctx["system"], ctx["device"]

    library = CompiledFunctionLibrary(**system.sindy_library_cfg())
    sindy = SINDyVectorized(library=library, n_out=system.nx, device=device)
    train_loader, dev_loader, _ = system.make_sysid_data(cfg, device)

    run_dir = new_run_dir(ctx["results_dir"] / "models" / "dynamics")
    logger = CustomLogger(
        args=None,
        savedir=str(run_dir / "logs"),
        verbosity=cfg.get("verbosity", 100),
        stdout=["train_loss", "dev_loss"],
    )
    train_sysid(system, sindy, train_loader, dev_loader, cfg, device, logger=logger)

    if cfg.get("debias", {}).get("enabled", False):
        support = [list(indices) for indices in sindy.active_idx]
        save_model(sindy, run_dir / "saved_models" / "sindy_pruned.pt")
        train_sysid(system, sindy, train_loader, dev_loader,
                    make_debias_config(cfg), device, logger=logger)
        if [list(indices) for indices in sindy.active_idx] != support:
            raise RuntimeError("System-ID debiasing changed the selected sparse support")
        save_model(sindy, run_dir / "saved_models" / "sindy_debiased.pt")

    model_path = run_dir / "saved_models" / "sindy.pt"
    save_model(sindy, model_path)
    snapshot_config(run_dir, cfg, ctx["seed"])
    print("\nIdentified dynamics:")
    sindy.pretty_print()
    print(f"\nSaved SINDy model to {model_path}")


def run_policy_training(system_name: str, case_dir: Path, policy_type: str) -> None:
    if policy_type not in {"sparse", "nn"}:
        raise ValueError("policy_type must be 'sparse' or 'nn'")
    label = "SD-DPC" if policy_type == "sparse" else "NN-DPC"
    _, ctx = _context(system_name, case_dir, "policy.yaml", f"{label} policy training")
    cfg, system, device = ctx["cfg"], ctx["system"], ctx["device"]
    if policy_type == "nn":
        # Both controllers share one policy configuration. NN-specific optimizer
        # settings override only the named values in the common configuration.
        cfg = deep_update(cfg, cfg.get("nn_training", {}))
    sindy = _load_sindy(ctx["results_dir"], cfg, device)

    if policy_type == "sparse":
        library = CompiledFunctionLibrary(**system.policy_library_cfg())
        policy = SINDyVectorized(
            library=library,
            n_out=system.nu,
            policy_name=[f"u{i}" for i in range(system.nu)],
            device=device,
            seed=cfg.get("policy_seed", 0),
        )
        init_scale = cfg.get("init_scale")
        if init_scale is not None:
            for coefficients in policy.Xi:
                coefficients.data.mul_(init_scale)
    else:
        policy = build_nn_policy(system, cfg.get("nn_policy", cfg)).to(device)

    train_loader, dev_loader = system.make_policy_data(cfg, device)
    run_dir = new_policy_run_dir(ctx["results_dir"], policy_type)
    logger = CustomLogger(
        args=None,
        savedir=str(run_dir / "logs"),
        verbosity=cfg.get("verbosity", 100),
        stdout=cfg.get("logger_metrics", ["train_loss", "dev_loss"]),
    )
    train_policy(
        system,
        sindy,
        policy,
        train_loader,
        dev_loader,
        cfg,
        device,
        logger=logger,
        action_scale=cfg.get("action_scale", 1.0),
    )

    if policy_type == "sparse":
        debias = cfg.get("debias", {})
        if debias.get("enabled", False):
            support = [list(indices) for indices in policy.active_idx]
            expected = cfg.get("expected_final_terms_per_output")
            if expected is not None and [len(indices) for indices in support] != expected:
                raise RuntimeError(
                    f"Sparse support {[len(indices) for indices in support]} "
                    f"differs from configured expected support {expected}; debiasing was not started"
                )
            save_model(policy, run_dir / "saved_models" / "policy_sparse_pruned.pt")
            debias_cfg = make_debias_config(cfg)
            train_policy(
                system, sindy, policy, train_loader, dev_loader, debias_cfg, device,
                logger=logger, action_scale=cfg.get("action_scale", 1.0),
            )
            if [list(indices) for indices in policy.active_idx] != support:
                raise RuntimeError("Debiasing changed the selected sparse support")
            save_model(policy, run_dir / "saved_models" / "policy_sparse_debiased.pt")
        model_path = run_dir / "saved_models" / "policy_sparse.pt"
        save_model(policy, model_path)
        print("\nLearned sparse policy:")
        policy.pretty_print()
    else:
        model_path = run_dir / "saved_models" / "policy_nn.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), model_path)
    snapshot_config(run_dir, cfg, ctx["seed"])
    print(f"\nSaved {label} policy to {model_path}")


def run_nn_distillation(system_name: str, case_dir: Path) -> None:
    """Fit NN-Distilled on the SD-DPC support and validate its reference metrics."""
    eval_name = "eval_nominal.yaml" if system_name == "twotank" else "eval.yaml"
    parser = argparse.ArgumentParser(
        description="Train and validate an NN-Distilled sparse policy",
    )
    parser.add_argument(
        "--config", type=Path, default=case_dir / "configs" / "policy.yaml",
        help="policy YAML containing the nn_distillation settings",
    )
    parser.add_argument(
        "--eval-config", type=Path, default=case_dir / "configs" / eval_name,
        help="scenario and safety configuration used for validation",
    )
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="write to this directory instead of a new numbered result run",
    )
    args = parser.parse_args()

    policy_cfg = load_config(args.config.expanduser().resolve())
    eval_cfg = load_config(args.eval_config.expanduser().resolve())
    settings = policy_cfg.get("nn_distillation")
    if not settings:
        raise ValueError(f"nn_distillation is missing from {args.config}")

    device = torch.device(args.device)
    seed_everything(int(settings["train_seed"]))
    system = make_system(system_name, device=device)
    results_dir = case_dir / "results"
    sparse_policy = _load_policy(results_dir, eval_cfg, device).eval()
    teacher = load_nn_policy(
        system, find_nn_checkpoint(results_dir, eval_cfg), eval_cfg, device=device,
    ).eval()

    print(
        f"[1/3] Collecting {settings['train_scenarios']} NN-DPC teacher rollouts...",
        flush=True,
    )
    training = make_distillation_scenarios(
        case_dir.name, system, eval_cfg,
        count=int(settings["train_scenarios"]),
        seed=int(settings["train_seed"]),
        horizon=int(settings["train_horizon"]),
        device=device,
    )
    policy, fit_report = distill_nn_policy(system, sparse_policy, teacher, training)

    print(
        f"[2/3] Validating on {settings['validation_scenarios']} held-out scenarios...",
        flush=True,
    )
    validation = make_distillation_scenarios(
        case_dir.name, system, eval_cfg,
        count=int(settings["validation_scenarios"]),
        seed=int(settings["validation_seed"]),
        horizon=int(settings["validation_horizon"]),
        device=device,
    )
    metrics = evaluate_distilled_policy(case_dir.name, system, policy, validation, eval_cfg)
    check = validate_reference_results(case_dir.name, metrics)
    if not check["passed"]:
        raise RuntimeError(
            "NN-Distilled validation does not match the reference results: "
            f"actual={check['actual']}, expected={check['expected']}"
        )

    if args.output_dir is None:
        run_dir = new_run_dir(results_dir / "models" / "policies" / "nn_distilled")
    else:
        run_dir = args.output_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "saved_models" / "policy_nn_distilled.pt"
    save_distilled_policy(model_path, policy)
    save_json(run_dir / "validation_metrics.json", metrics)
    save_json(run_dir / "fit_report.json", fit_report)
    snapshot_config(run_dir, {**policy_cfg, "evaluation": eval_cfg}, int(settings["train_seed"]))
    print("[3/3] Table task, effort, violation-rate, and memory values matched.", flush=True)
    print(f"Saved NN-Distilled policy to {model_path}")
    print("\nRetained terms:")
    for output, support in enumerate(policy.active_idx):
        terms = [sparse_policy.function_names[index] for index in support]
        print(f"  u{output}: {', '.join(terms)}")
    print("\nValidation metrics:")
    print(f"  task loss: {metrics['task_loss_mean']:.12g} +- {metrics['task_loss_std']:.12g}")
    print(
        "  control effort: "
        f"{metrics['control_effort_mean']:.12g} +- {metrics['control_effort_std']:.12g}"
    )
    print(f"  violating scenarios: {metrics['scenario_violation_percent']:.1f}%")
    print(f"  indexed policy memory: {metrics['policy_sparse_indexed_kib']:.6f} KiB")


def run_evaluation(system_name: str, case_dir: Path) -> None:
    if system_name == "twotank":
        _, ctx = _context(system_name, case_dir, "eval_nominal.yaml", "Nominal controller evaluation")
        cfg, system, device = ctx["cfg"], ctx["system"], ctx["device"]
        policy = _load_policy(ctx["results_dir"], cfg, device)
        nn_policy = load_nn_policy(
            system, find_nn_checkpoint(ctx["results_dir"], cfg), cfg, device=device,
        )
        output = evaluate_twotank_nominal(system, policy, nn_policy, cfg, device)
        run_dir = new_run_dir(ctx["results_dir"] / "eval" / "nominal")
        save_json(run_dir / "metrics_summary.json", output["summary"])
        save_json(run_dir / "violation_summary.json", output["violations"])
        torch.save(
            {"trajectories": output["trajectories"], "scenarios": output["scenarios"]},
            run_dir / "all_trajectories.pt",
        )
        snapshot_config(run_dir, cfg, ctx["seed"])
        print(f"Saved nominal Two Tank evaluation to {run_dir}")
        return
    _, ctx = _context(system_name, case_dir, "eval.yaml", "Controller evaluation")
    cfg, system, device = ctx["cfg"], ctx["system"], ctx["device"]
    sindy = _load_sindy(ctx["results_dir"], cfg, device)
    policy = _load_policy(ctx["results_dir"], cfg, device)
    nn_policy = (
        load_nn_policy(
            system,
            find_nn_checkpoint(ctx["results_dir"], cfg),
            cfg,
            device=device,
        )
        if "nn_dpc" in cfg.get("methods", [])
        else None
    )

    run_dir = new_run_dir(ctx["results_dir"] / "eval")
    output = evaluate(
        system,
        sindy,
        policy,
        cfg,
        device,
        out_dir=run_dir,
        nn_policy=nn_policy,
    )
    snapshot_config(run_dir, cfg, ctx["seed"])
    print("\n" + format_metrics_table(output["summary"]))
    print(f"\nSaved evaluation to {run_dir}")
