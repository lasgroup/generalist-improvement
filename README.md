# Reinforcement Learning Enables Scaling Generalist Robot Policy Improvement

This repo contains a minimal implementation of multi-task post-training for π0.5 on LIBERO-90 and MolmoSpaces, using filtered SFT (FSFT), Best-of-N (BoN) or OGPO.

## Setup

### 1. Clone

```bash
git clone --recurse-submodules <repo-url> generalist-pi
```

### 2. Build the container

Everything runs inside the container described by [docker/generalist-pi.Dockerfile](docker/generalist-pi.Dockerfile).

```bash
cd generalist-pi
docker build . -f docker/generalist-pi.Dockerfile -t generalist-pi
```

To start an interactive shell in the container, run this from the repo root. The repo must be mounted at `/app`.

```bash
docker run --rm -it --gpus all --network=host -v "$(pwd)":/app generalist-pi bash
```

The code requires setting these global variables:

```
OPENPI_DATA_HOME = "/path/to/openpi_data_home"
MLSPACES_ASSETS_DIR = "/path/to/molmospaces/assets"
HF_HOME = "/path/to/hf_home"
HF_TOKEN = "hf_..."
WANDB_API_KEY = "..."
MUJOCO_GL = "egl"
XLA_PYTHON_CLIENT_MEM_FRACTION = "0.9"
```

The code also requires installing assets through assets through

  `uv run scripts/install_molmo_assets.py`.

The remaining assets are downloaded automatically.

### 3. Launch

Before launching, edit hard-coded values for your cluster and allocation at the top of [scripts/launcher.py](scripts/launcher.py).
Then, simply:

```bash
uv run scripts/launcher.py --config scripts/configs/multitask/bofn/molmo/bofn_molmo_tasks64_v0.yaml --mode local 
```

Each job requires one 4-GPU node with ~100GB VRAM each.

Note: the repo contains forks of `openpi` and `molmospaces`, which apply minimal changes on top of commits `981483dca0fd9acba698fea00aa6e52d56a66c58` and `7a2612cccdc3a7733d777b9cea1ac387d0a3db82`, respectively.