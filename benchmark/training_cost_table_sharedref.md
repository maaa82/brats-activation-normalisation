# Training cost per activation, basic nnU-Net (IN, batch 2), fold 0, real epochs

GPU: **NVIDIA A40**, 96 vCPUs. Epoch = 250 training + 50 validation iterations, patch 128³, AMP, deep supervision, nnU-Net default dataloader/augmentation. Epoch 0 (warm-up) excluded. Peak memory = `torch.cuda.max_memory_allocated` over the epoch (full process: model, activations, optimiser state); reserved = allocator high-water mark.

## Wave-normalised (use this table)

Every activation is normalised against **LeakyReLU** measured in its own five-way wave, so concurrent-load differences between waves cancel. This matters: the shared reference itself measured 206.2 / 213.1 / 216.0 s in waves A / B / C — a 4.8% drift, the same magnitude as the entire activation-to-activation spread. Raw epoch times from different waves must not be compared directly.

| Activation | waves | rel. to ref | norm. epoch (s) | SD% | peak alloc (GB) | model-only (GB) | rel. mem | GPU-h / 500-ep fold |
|---|---|---|---|---|---|---|---|---|
| Smish | C | 0.975 | 206.4 | 6.3% | 15.41 | 15.262 | 3.00 | 28.7 |
| HardELiSH | B | 0.977 | 206.8 | 5.0% | 8.98 | 8.832 | 1.75 | 28.7 |
| Logish | C | 0.979 | 207.4 | 2.5% | 15.41 | 15.261 | 3.00 | 28.8 |
| Swish | B | 0.98 | 207.4 | 4.2% | 6.29 | 6.14 | 1.23 | 28.8 |
| TanhExp | C | 0.991 | 209.8 | 5.7% | 13.98 | 13.832 | 2.73 | 29.1 |
| ReLU | AC | 0.998 | 211.4 | 4.2% | 5.13 | 4.98 | 1.00 | 29.4 |
| ELiSH | B | 0.998 | 211.3 | 7.3% | 8.98 | 8.832 | 1.75 | 29.3 |
| LeakyReLU | ABC | 1.0 | 211.7 | 4.9% | 5.13 | 4.98 | 1.00 | 29.4 |
| Mish | B | 1.002 | 212.1 | 6.9% | 13.98 | 13.833 | 2.73 | 29.5 |
| PReLU | A | 1.007 | 213.1 | 4.5% | 6.12 | 5.973 | 1.19 | 29.6 |
| ELU | A | 1.014 | 214.7 | 3.7% | 5.13 | 4.98 | 1.00 | 29.8 |
| GELU | A | 1.014 | 214.6 | 4.4% | 6.12 | 5.973 | 1.19 | 29.8 |

Reference (`LeakyReLU`) grand mean = **211.7 s**.

**Timing.** Full spread across 12 activations is 4.0%, against a mean per-activation epoch-to-epoch SD of 5.0%. No activation is distinguishable from any other in training time: at batch 2 with five concurrent trainings, epoch time is set by dataloading, augmentation and network-filesystem reads, not by the non-linearity.

**Normalisation residual (validity check).** Activations measured in more than one wave should give the same normalised value in each; the discrepancy is the residual uncertainty that survives normalisation.

- `ReLU`: A=1.016, C=0.981 → 3.6% residual

Residual is ~±1.8%. Differences in the table smaller than this are noise: quote the range, not the ordering.

**Memory.** Reproducible to ~0.01 GB across independent runs and unaffected by wave composition, so the memory ranking is reliable even though the timing ranking is not. Differences come from what autograd must retain: `inplace=True` primitives keep nothing, while activations written as chains of primitive ops save an intermediate per link over the full patch 128³ × batch 2 activation map.

**Model-only column** is `vram_profile_grid.py --grid in-only` (architecture + activation, no optimiser state or dataloader). The real-epoch peak sits +0.148 GB above it, with a spread of only 0.003 GB across all activations, so the model-only figure is a faithful proxy for the real footprint.

## Raw per-activation (latest run each; NOT normalised)

Kept for traceability. The `rel.` column here compares runs that may come from different waves and is unreliable whenever the twelve were not all trained in a single wave — prefer the table above.

| Activation | epochs | epoch time (s) mean ± SD | rel. | peak alloc (GB) | peak reserved (GB) | rel. mem | GPU-h / 500-epoch fold |
|---|---|---|---|---|---|---|---|
| PReLU | 9 | 207.5 ± 9.3 | 0.961 | 6.12 | 7.36 | 1.195 | 28.8 |
| HardELiSH | 9 | 208.2 ± 10.4 | 0.964 | 8.98 | 9.73 | 1.754 | 28.9 |
| Swish | 9 | 208.7 ± 8.8 | 0.966 | 6.29 | 7.63 | 1.229 | 29.0 |
| ELU | 9 | 209.0 ± 7.7 | 0.968 | 5.13 | 6.46 | 1.002 | 29.0 |
| GELU | 9 | 209.0 ± 9.2 | 0.968 | 6.12 | 6.58 | 1.195 | 29.0 |
| Smish | 9 | 210.5 ± 13.4 | 0.975 | 15.41 | 15.94 | 3.01 | 29.2 |
| Logish | 9 | 211.5 ± 5.3 | 0.979 | 15.41 | 15.93 | 3.01 | 29.4 |
| ReLU | 9 | 211.8 ± 6.6 | 0.981 | 5.13 | 6.4 | 1.002 | 29.4 |
| ELiSH | 9 | 212.6 ± 15.5 | 0.984 | 8.98 | 9.72 | 1.754 | 29.5 |
| Mish | 9 | 213.5 ± 14.8 | 0.988 | 13.98 | 14.5 | 2.73 | 29.6 |
| TanhExp | 9 | 214.0 ± 12.2 | 0.991 | 13.98 | 14.49 | 2.73 | 29.7 |
| LeakyReLU | 9 | 216.0 ± 13.1 | 1.0 | 5.12 | 6.4 | 1.0 | 30.0 |

## Production batch-8 runs (500 real epochs, 5 folds) — same GPU

| model | fold | epochs | median epoch (s) | mean epoch (s) | GPU-h |
|---|---|---|---|---|---|
| BN_LeakyReLU_bs8 | 0 | 536 | 52.1 | 52.0 | 7.74 |
| BN_LeakyReLU_bs8 | 1 | 536 | 52.0 | 52.2 | 7.77 |
| BN_LeakyReLU_bs8 | 2 | 536 | 52.1 | 52.1 | 7.76 |
| BN_LeakyReLU_bs8 | 3 | 535 | 52.1 | 52.1 | 7.74 |
| BN_LeakyReLU_bs8 | 4 | 536 | 52.1 | 52.2 | 7.77 |
| GN_LeakyReLU_bs8 | 0 | 500 | 50.4 | 50.1 | 6.97 |
| GN_LeakyReLU_bs8 | 1 | 500 | 50.0 | 50.2 | 6.97 |
| GN_LeakyReLU_bs8 | 2 | 500 | 50.3 | 50.1 | 6.95 |
| GN_LeakyReLU_bs8 | 3 | 500 | 50.5 | 50.2 | 6.98 |
| GN_LeakyReLU_bs8 | 4 | 500 | 50.0 | 50.2 | 6.98 |

## Caveats

- Five trainings share the pod's 96 vCPUs during every wave, as in production, so CPU-side augmentation and network-storage reads are included in epoch time. Runs measured under a different number of concurrent trainings are not comparable — this is what the wave normalisation corrects for.
- Memory is the real process footprint at batch 2 (model, activations, optimiser state), not model-only.
- fold 0 only; activation cost does not depend on the fold.
- `GPU-h / 500-ep fold` extrapolates the epoch time linearly to 500 epochs and excludes the end-of-training validation prediction, which these benchmark runs skip.
