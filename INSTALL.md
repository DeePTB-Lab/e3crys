# Installation

This project uses `uv` for reproducible dependency resolution.

## CPU environment

The checked-in `pyproject.toml` and `uv.lock` mirror the existing
`requirements.txt` CPU stack, using the same PyG installation pattern as
DeePTB:

- PyTorch `2.11.0`
- PyG `2.7.0`
- `torch-scatter==2.1.2`
- `torch-cluster==1.6.3`
- `torch-sparse==0.6.18`
- `pyg-lib==0.6.0`
- PyG CPU wheels from `https://data.pyg.org/whl/torch-2.11.0+cpu.html`

Create the environment with:

```bash
uv sync --locked
```

Run the main entry point with:

```bash
uv run e3crys --help
```

## PyG compatibility note

PyG compiled extensions must match the PyTorch build and platform. This
project provides PyG's CPU wheel page through `tool.uv.find-links` and uses
platform markers for PyG extension packages. macOS wheels use versions such as
`torch-scatter==2.1.2`, while Linux/Windows CPU wheels use local-version
variants such as `torch-scatter==2.1.2+pt211cpu`.

Keeping these variants explicit makes the lock file usable both on local macOS
development machines and in the Linux CI image.

For CUDA training, create a separate lock/configuration that aligns all of
these versions together:

- CUDA runtime version
- PyTorch build
- `torch-geometric`
- `torch-scatter`
- `torch-cluster`
- `torch-sparse`
- `pyg-lib`

Do not mix PyTorch and PyG extension builds from different Torch/CUDA targets.
