# Two Tank

Continuous two-tank dynamics identification and constrained level-tracking policy learning. `configs/sysid.yaml` and `configs/policy.yaml` follow paper Table I and enable fixed-support debiasing. The included pretrained checkpoints are in `checkpoints/`, while their recorded policy settings are retained in `configs/policy_recorded_run.yaml`. Use `notebooks/00_colab_workflow.ipynb` for the combined Colab tutorial or `01_system_id.ipynb` through `04_plot_evaluation_results.ipynb` for separate stages; matching commands are under `scripts/TwoTank/`.
