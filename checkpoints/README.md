# Trained Policy Checkpoints

Trained NumPy PPO checkpoints (`*.npz`) are saved here by `scripts/train_ppo.py` using the naming convention:

- `H_seed0.npz` … `H_seed9.npz` (full 6-option hierarchical orchestrator)
- `H-NU_seed0.npz` … `H-NU_seed9.npz` (3-option UAV-free ablation)
- `F_seed0.npz` … `F_seed2.npz` and `F+S_seed0.npz` … `F+S_seed2.npz` (91-action flat PPO)
- `H-LN_seed0.npz` … `H-LN_seed2.npz` and `H-NS_seed0.npz` … `H-NS_seed2.npz`

Upload your trained `.npz` checkpoints from your experiment run into this folder before tagging a Zenodo release.
