# diff-cam

A differentiable CNC simulator based on Taichi.

## Setup

- Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync
```

- Install TurboVNC server:

```bash
./scripts/setup.sh
```

## Documentation

See [docs/design.md](docs/design.md) for design details.

## PPO Training

TODO:

- Create cam_env for the simulator (currently in `main.py`)
- Register the cam_env with gymnasium
- Load the env with pufferlib's vectorized loader inside of `ppo.py`
- Train with:

```bash
./algorithms/ppo.py
```

## AMD GPU Support

Taichi support for AMD GPUs may be possible via the ROCm: see <https://rocm.docs.amd.com/projects/taichi/en/latest/install/taichi-install.html>

## Running Tests with Pytest

```bash
uv run pytest
```
