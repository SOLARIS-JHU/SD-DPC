# Case-study commands

Run these from the repository root. Each case has `system_id.py`, `train_sd_dpc.py`, `train_nn_dpc.py`, and `evaluation.py` under `scripts/<Case>/`.

```bash
python scripts/DoubleIntegrator/system_id.py
python scripts/DoubleIntegrator/train_sd_dpc.py
python scripts/DoubleIntegrator/train_nn_dpc.py
python scripts/DoubleIntegrator/evaluation.py
```

Use `TwoTank` or `VanDerPol` in place of `DoubleIntegrator` for the other paper cases. VanDerPol is the one-input, relative-degree-two system. All scripts accept `--config`, `--device`, and `--seed`. The default YAML files live under `case_studies/<Case>/configs/`.

System identification and sparse policy training use the paper-aligned YAML defaults and run their configured fixed-support debias phases. Evaluation scripts compare the included publication checkpoints; the saved paper figures and controller table are in `results/`. The complete installation and rerun tutorial is in the repository's main `README.md`.
