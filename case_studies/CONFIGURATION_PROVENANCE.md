# Configuration provenance

The default training YAML files follow Table I of the paper. They are the authoritative settings for new system-identification, SD-DPC, and NN-DPC runs.

The included final checkpoints predate that normalization. Their original sparse-policy settings are preserved in `configs/policy_recorded_run.yaml` and `configs/policy_final_run.json` within each case directory. Evaluation YAML files reconstruct the neural checkpoint with its recorded GELU activation so that the stored state dictionaries remain loadable.

The main recorded-policy differences from Table I are:

| Case | Paper-aligned default | Included checkpoint record |
| --- | --- | --- |
| Two Tank | 1000 scenarios; leak 0.05; band 0.1; $Q_r=Q_x=1$; 1000 debias epochs | 2000 scenarios; leak 0.01; band 0.05; $Q_r=Q_x=20$; 800 debias epochs |
| Van der Pol | $\lambda_\pi=0.002$; $\tau_\pi=0.125$; leak/band 0.1; $Q_{dx}=1$; 1000 debias epochs | $\lambda_\pi=0.001$; $\tau_\pi=0.0625$; leak 0.01; band 0.05; $Q_{dx}=0$; 800 debias epochs |
| 2-D Integrator | leak 0.1; band 0.5; 1000 debias epochs | leak 0.05; band 0.01; 800 debias epochs |

For this reason, a new run using the default paper-aligned YAML is not claimed to reproduce the exact coefficients stored in the included checkpoints. The recorded files are retained for transparent artifact provenance and checkpoint evaluation.

## Publication checkpoint verification

The checked-in files were compared byte-for-byte with the final paper assets. `dynamics_sindy.pt` matches `Latest_SparseDPC_Final_Runs/assets/dynamics/`; `policy_sparse.pt` matches both the selected final-run checkpoint and `results_ACC/<Case>/selected_policy/policy_sparse.pt`; and `policy_nn.pth` matches the NN-DPC asset used by the paper evaluation.

| Case | Artifact | SHA-256 |
| --- | --- | --- |
| Two Tank | `dynamics_sindy.pt` | `3ba9b0cb9f893bb95860e043dfb2c48d8e10c11dd878fe6f97f3f32c32e741be` |
| Two Tank | `policy_sparse.pt` | `12158876fbdc50b035b63be03571a27d341f0faedcc57a75c420702ea1f33254` |
| Two Tank | `policy_nn.pth` | `5dc078b0373a66f2e2f41dc218db8dcd5e7ae4f4f0ba86c90815df7d2d5aa707` |
| Van der Pol | `dynamics_sindy.pt` | `5ef20ca8a161b113848a291f1775601ffc170a143fb61a7577cb48915fe2c49b` |
| Van der Pol | `policy_sparse.pt` | `fdd2f3d3b306eb77593430bbc854c9cb26e4b6cf80bc3fe04fdcd23708e0ad07` |
| Van der Pol | `policy_nn.pth` | `93be1179da9cbf6e23da94bf42660b36e1061ac200478a80204c38a5f8c999ad` |
| 2-D Integrator | `dynamics_sindy.pt` | `da07d43eeae2df62eb690c8b8b8ab0e69c9179ca192446329acccd2d69250f38` |
| 2-D Integrator | `policy_sparse.pt` | `d0cb63f175598db9e746f7d7ea23966eb17673b61bf463d44a4dc6ee9bd0c2e7` |
| 2-D Integrator | `policy_nn.pth` | `bc759cabd1d8847516e72ba5c819108cef55462e341f00092cab429b3716805a` |
