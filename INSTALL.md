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
project follows DeePTB's approach for PyG extension packages: avoid hard-coding
Linux/Windows local-version suffixes such as `+pt211cpu` for every platform,
and provide PyG's CPU wheel page through `tool.uv.find-links`. That lets `uv`
select platform-appropriate wheels for macOS, Linux, or Windows.

`pyg-lib` is the exception: PyG publishes macOS arm64 wheels as `0.6.0+pt211`
and Linux/Windows CPU wheels as `0.6.0+pt211cpu`, so `pyproject.toml` uses
platform markers for that package.

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
