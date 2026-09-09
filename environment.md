# Environment

Exact versions used for the nnU-Net / BraTS 2023 experiments in this repository.

| Component | Version |
|---|---|
| Python | 3.11 |
| nnU-Net | v2.2.1 (pinned) |
| PyTorch | 2.4.0 |
| CUDA | 12.4 |
| numpy | < 2.1 |

## Hardware

- **This repository:** NVIDIA A40, 46 GB, one GPU per training. Every cell
  archived here — the activation screen, the three normalisation
  configurations, the batch-size cells and the control cell — was trained on
  this hardware.
- **Companion study (published separately):** ran on Google Colab A100. Its
  code and results are not part of this repository — see the paper linked in
  the README.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install "numpy<2.1"
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install nnunetv2==2.2.1
```

nnU-Net expects the usual three environment variables:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

## Notes

- `numpy < 2.1` is required: nnU-Net v2.2.1 and the pinned PyTorch build were
  not validated against the numpy 2.1 ABI.
- The trainer files in `trainers/` target the nnU-Net v2.2.1 API specifically.
  In this version `build_network_architecture` is called on the trainer
  **class** rather than on an instance, which is why the BN, GN and batch-size
  trainers define it as a `@classmethod`. Porting these files to a different
  nnU-Net version requires re-checking that call convention.
