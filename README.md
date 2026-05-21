# High Order

High Order is a research codebase for high-order invariant and equivariant graph
neural networks on crystalline materials. It supports self-supervised training,
scalar property prediction, and tensor property prediction.

The main modeling path combines atom and distance embeddings, invariant message
passing, equivariant tensor-product updates, and readout layers for scalar or
Voigt-form tensor outputs.

## Repository Layout

- `main.py`: command-line entry point.
- `src/model/`: embedding, invariant/equivariant layers, tensor products, MLPs,
  readouts, and e3nn utilities.
- `src/train_test/`: training, validation, testing, checkpointing, metrics, and
  visualization.
- `data/`: dataset definitions, dataloaders, property lists, and processed
  tensor datasets.
- `tests/`: current smoke tests for the maintained model path.

Required model constants are kept in:

- `src/model/Jd.pt`
- `src/model/z_rot_indices_lmax12.pt`

## Installation

This project uses `uv`.

```bash
uv sync --locked
```

The lock file targets the CPU PyTorch/PyG stack:

- PyTorch `2.11.0`
- PyG `2.7.0`
- PyG extension wheels from `https://data.pyg.org/whl/torch-2.11.0+cpu.html`

See [INSTALL.md](INSTALL.md) for platform notes and CUDA guidance.

## Usage

Show available options:

```bash
uv run e3crys --help
```

The historical entry point remains available:

```bash
uv run python main.py --help
```

Run a tensor-only smoke-sized training invocation by limiting epochs yourself,
for example:

```bash
uv run e3crys --no-need-self-train --no-need-scalar-train --tensor-train-limit 1 --tensor-num-epochs 1
```

Full training writes checkpoints, metrics, TensorBoard logs, and figures to the
configured output directories.

## Tests

Run the maintained smoke tests:

```bash
uv run pytest
```

`pytest` is configured to collect tests only from `tests/`. Older scripts under
`src/model/utils/` are not treated as the maintained test suite.

GitHub Actions runs the same CLI check and test suite on Ubuntu. The CI image
workflow publishes `ghcr.io/deeptb-lab/e3crys/ci:py3.12-cpu` so repeated CI
runs can reuse the locked CPU dependency stack instead of rebuilding the full
environment from scratch.

## Data

The repository currently includes processed tensor datasets such as dielectric,
elastic, and piezoelectric JARVIS-derived pickle files. Some paths in
`data/dataloaders/name_path.json` point to external or local datasets that are
not committed, such as Materials Project, Alexandria, and several scalar
property raw datasets.

`pretrained/*.pth` is reserved for collaboration artifacts that help reproduce
shared results. Other checkpoints, generated neighbor-list caches, training
outputs, local databases, or temporary logs should stay out of Git.
