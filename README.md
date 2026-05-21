# e3crys

`e3crys` is a research codebase for invariant and equivariant graph neural
networks on crystalline materials. The project supports self-supervised
pretraining, scalar property prediction, and Voigt-form tensor property
prediction.

The main modeling path combines atom and distance embeddings, invariant message
passing, equivariant tensor-product updates, and readout layers for scalar or
tensor outputs.

This repository is being prepared as the DeePTB-Lab organization version of the
project. The command-line entry point is `e3crys`. The Python distribution name
is still `high-order-new` during this migration and should be renamed in a
separate packaging change.

## Repository Layout

- `main.py`: command-line entry point and training orchestration.
- `src/model/`: embedding layers, invariant/equivariant layers, tensor
  products, readouts, MLPs, GMTNet components, and e3nn utilities.
- `src/train_test/`: training, validation, testing, checkpoint, metric, and
  visualization helpers.
- `data/`: dataset definitions, dataloaders, property lists, processed JARVIS
  tensor datasets, and exploration notebooks.
- `tests/`: maintained smoke tests for the currently supported model path.

Required model constants are tracked in Git:

- `src/model/Jd.pt`
- `src/model/z_rot_indices_lmax12.pt`

## Installation

This project uses `uv` and a checked-in `uv.lock` for reproducible installs.

```bash
uv sync --locked
```

Then verify the command-line entry point:

```bash
uv run e3crys --help
```

The lock file targets Python `>=3.10,<3.13` and the current PyTorch/PyG stack:

- PyTorch `2.11.0`
- PyG `2.7.0`
- `pyg-lib`, `torch-cluster`, `torch-scatter`, and `torch-sparse` from the PyG
  wheel index for `torch-2.11.0+cpu`

PyG compiled extensions are platform-specific. The lock file includes macOS
wheels such as `torch-scatter==2.1.2` and Linux/Windows CPU wheel variants such
as `torch-scatter==2.1.2+pt211cpu`.

See [INSTALL.md](INSTALL.md) for platform notes and CUDA guidance.

## Usage

Show available options:

```bash
uv run e3crys --help
```

The historical script entry point remains available:

```bash
uv run python main.py --help
```

Run a tensor-only smoke-sized training invocation by limiting epochs and sample
count yourself:

```bash
uv run e3crys --no-need-self-train --no-need-scalar-train --tensor-train-limit 1 --tensor-num-epochs 1
```

Full training writes checkpoints, metrics, TensorBoard logs, and figures to the
configured output directories. Generated training outputs should not be
committed.

## Tests and CI

Run the maintained smoke tests:

```bash
uv run pytest
```

GitHub Actions runs the same CLI check and test suite on Ubuntu. A separate
workflow builds and publishes the dependency image:

```text
ghcr.io/deeptb-lab/e3crys/ci:py3.12-cpu
```

The tag name reflects the intended CPU PyG extension stack. On Linux, upstream
PyTorch wheels may still bring runtime packages published with the PyTorch wheel
itself, so treat the image as the locked CI dependency image rather than a
minimal CPU-only runtime image.

The latest workflow status should be checked in GitHub Actions after changes to
`pyproject.toml`, `uv.lock`, `Dockerfile.ci`, or workflow files.

## Data and Artifacts

The repository intentionally keeps:

- `data/explore.ipynb`
- processed JARVIS tensor datasets in `data/*.pkl`
- `pretrained/self_train_epochs100_epoch100.pth` for collaboration and result
  reproduction

Some paths in `data/dataloaders/name_path.json` point to external or local
datasets that are not committed, including Materials Project, Alexandria, and
several raw scalar-property datasets.

Do not commit generated neighbor-list caches, local databases, checkpoints
outside the agreed `pretrained/` artifact, metrics, figures, TensorBoard logs,
or temporary runtime files.
