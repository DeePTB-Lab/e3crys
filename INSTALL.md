# Installation

This project uses `uv` for reproducible dependency resolution. The checked-in
`pyproject.toml` describes the supported Python package environment, and
`uv.lock` records the exact resolved dependencies used by local development and
CI.

## Supported Baseline

- Python `>=3.10,<3.13`
- PyTorch `2.11.0`
- PyG `2.7.0`
- PyG extension wheels from
  `https://data.pyg.org/whl/torch-2.11.0+cpu.html`
- Python distribution name and command-line entry point: `e3crys`

Create the environment with:

```bash
uv sync --locked
```

Check the CLI:

```bash
uv run e3crys --help
```

Run tests:

```bash
uv run pytest
```

## PyG Platform Notes

PyG compiled extensions must match the PyTorch build and platform. The project
uses `tool.uv.find-links` to point `uv` at the PyG wheel page and uses explicit
platform markers for compiled PyG packages:

- macOS uses versions such as `torch-scatter==2.1.2`
- Linux and Windows use CPU local-version variants such as
  `torch-scatter==2.1.2+pt211cpu`

The same pattern is used for `torch-cluster`, `torch-sparse`, and `pyg-lib`.
Keeping these variants explicit prevents a lock file generated on macOS from
forcing Linux CI or Linux users onto macOS-only wheels.

## CI Image

GitHub Actions builds and publishes:

```text
ghcr.io/deeptb-lab/e3crys/ci:py3.12-cpu
```

The image pre-installs the locked project dependency environment under
`/opt/e3crys/.venv`. CI then mounts the repository at `/workspace`, runs
`uv sync --locked`, checks `uv run e3crys --help`, and runs `uv run pytest`.

The tag name reflects the intended CPU PyG extension stack. It is not currently
an aggressively minimized runtime image, and Linux PyTorch wheels may include
additional runtime packages from PyTorch's own distribution.

## CUDA Guidance

CUDA training should use a separate lock/configuration that aligns all compiled
packages together:

- CUDA runtime version
- PyTorch build
- `torch-geometric`
- `torch-scatter`
- `torch-cluster`
- `torch-sparse`
- `pyg-lib`

Do not mix PyTorch and PyG extension builds from different Torch or CUDA
targets.
