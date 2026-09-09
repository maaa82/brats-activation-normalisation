# Per-case Synapse validation metrics

One CSV per trained model. Each file is the official per-case scoring returned by
the BraTS 2023 Adult Glioma continuous-evaluation queue on Synapse for a
five-fold soft-voting ensemble prediction of that model.

## Rows

Each file contains **219 case rows** — the blinded BraTS 2023 adult glioma
validation cases — followed by **5 summary rows** written by the Synapse scorer,
labelled `mean`, `std`, `25quantile`, `median`, `75quantile`. Drop the summary
rows before any per-case statistics; recompute summaries from the 219 case rows
rather than reading the summary rows, so that the treatment of missing lesions
is explicit in your own code. All 23 files have been checked and contain
exactly 219 case rows.

## Columns

The first column is unnamed and holds the case identifier
(`BraTS-GLI-XXXXX-000`). The remaining 15 are:

| Column | Meaning |
|---|---|
| `LesionWise_Dice_ET` / `_TC` / `_WT` | Lesion-wise Dice similarity coefficient for Enhancing Tumour, Tumour Core and Whole Tumour. Higher is better (0–1). |
| `LesionWise_Hausdorff95_ET` / `_TC` / `_WT` | Lesion-wise 95th-percentile Hausdorff distance in millimetres for each region. Lower is better. |
| `Num_TP_ET` / `_TC` / `_WT` | Number of true-positive lesions matched for that region in that case. |
| `Num_FP_ET` / `_TC` / `_WT` | Number of false-positive lesion components predicted for that region. |
| `Num_FN_ET` / `_TC` / `_WT` | Number of ground-truth lesions for that region that were missed. |

Regions follow the BraTS composite definitions: **ET** = enhancing tumour
(label 3), **TC** = tumour core (necrotic core + enhancing tumour), **WT** =
whole tumour (all foreground labels).

These are the **official BraTS 2023 lesion-wise metrics**, computed by the
challenge organisers against sequestered ground-truth labels that are not
publicly available. They are not re-implementations, and they are not
recomputable locally.

Under the lesion-wise protocol a missed lesion is penalised per lesion rather
than per voxel: a false negative contributes a Dice of 0 and a Hausdorff
distance of 374 mm for that lesion, which is why HD95 means are large and
heavy-tailed compared with legacy per-case metrics.

## Files

23 files, one per trained cell.

| Files | Configuration |
|---|---|
| `IN_LeakyReLU.csv`, `IN_ReLU.csv`, `IN_PReLU.csv`, `IN_Swish.csv`, `IN_TanhExp.csv`, `IN_ELU.csv`, `IN_GELU.csv`, `IN_Mish.csv`, `IN_ELiSH.csv`, `IN_HardELiSH.csv`, `IN_Logish.csv`, `IN_Smish.csv` | Twelve-activation in-domain screen — Instance Norm (nnU-Net default), batch 2 |
| `BN_LeakyReLU.csv`, `BN_PReLU.csv`, `BN_Swish.csv`, `BN_TanhExp.csv` | Batch Norm, batch 2 |
| `GN_LeakyReLU.csv`, `GN_PReLU.csv`, `GN_Swish.csv`, `GN_TanhExp.csv` | Group Norm (8 groups), batch 2 |
| `BN_LeakyReLU_bs8.csv`, `GN_LeakyReLU_bs8.csv` | Batch-size experiment, batch 8 |
| `BN_PReLU_frozen.csv` | Ablation control: BN affine trainable, PReLU slope frozen at 0.25 |

The four Instance Norm cells of the 3 × 4 factorial (LeakyReLU, PReLU, Swish,
TanhExp) are the same trainings as the corresponding cells of the screen; they
are not duplicated.
