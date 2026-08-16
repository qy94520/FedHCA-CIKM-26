# FedHCA

Official implementation for our CIKM 2026 paper: FedHCA: Hierarchical Contribution-Aware Aggregation for Asynchronous Federated Class-Incremental Learning

## Contents

- `src/fl_main.py`: main experiment entry point.
- `src/FedHCA.py`: client-side training and continual-learning components.
- `src/Fed_utils.py`: federated aggregation, evaluation, and utility functions.
- `src/iCIFAR100.py`: CIFAR-100, CIFAR-10, and SVHN incremental dataset wrappers.
- `src/ResNet.py`, `src/myNetwork.py`, `src/backbone_factory.py`: model definitions.

Large assets are intentionally excluded: datasets, pretrained checkpoints, logs, cached files, and generated figures.

## Setup

```bash
pip install -r requirements.txt
```

Place datasets under `data/` or pass the dataset root using the corresponding command-line argument.

## Example

```bash
bash scripts/run_cifar100.sh
```

The script writes checkpoints, logs, and CSV metrics under `outputs/`.

## Notes

This public version keeps only the code needed to reproduce the main CIFAR/SVHN experiments. Local paths, temporary outputs, cached bytecode, trained weights, and environment-specific launch scripts have been removed.
