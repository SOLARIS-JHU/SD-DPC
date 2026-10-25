# Van der Pol

A one-input, relative-degree-two oscillator for sparse dynamics identification and policy learning. `configs/policy.yaml` contains the sparse-policy, NN-DPC, and NN-Distilled settings. The pretrained checkpoints are in `checkpoints/`. Use `notebooks/00_colab_workflow.ipynb` for the combined Colab tutorial or `01_system_id.ipynb` through `04_plot_evaluation_results.ipynb` for separate stages; matching commands, including `train_nn_distilled.py`, are under `scripts/VanDerPol/`.
