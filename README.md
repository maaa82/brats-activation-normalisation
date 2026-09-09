# Activation function × normalisation strategy in nnU-Net for BraTS 2023

Code and per-case results for a factorial study of **activation functions ×
normalisation strategies** in nnU-Net on the BraTS 2023 adult glioma cohort.
The study trains nnU-Net's default `PlainConvUNet` under three normalisation
strategies — Instance Norm (nnU-Net's default), Batch Norm and Group Norm
(8 groups) — crossed with four activation functions (LeakyReLU, PReLU,
Swish/SiLU, TanhExp), together with a twelve-activation in-domain screen under
Instance Norm, a batch-size experiment at batch 8, and an ablation isolating
the effect of a frozen PReLU slope. Every model is evaluated on the 219
blinded BraTS 2023 validation cases using the official lesion-wise metrics
returned by the Synapse continuous-evaluation queue. This repository contains
the trainer definitions, the per-case metric files, the fold splits, the
training-cost benchmark and the analysis notebooks; it does not contain imaging
data or model weights.

- **Accompanying paper:** *Activation function × normalisation strategy in
  nnU-Net for BraTS 2023 brain tumour segmentation* — in review. The citation
  and DOI will be added here on publication.
- **Companion paper (published):** *Informatics* 2026,
  [doi:10.3390/informatics13070118](https://doi.org/10.3390/informatics13070118).
  That earlier study screens activation functions on a custom U-Net; its code
  and results live with that paper and are not reproduced here.

---

## 1. Environment

| Component | Version |
|---|---|
| Python | 3.11 |
| nnU-Net | v2.2.1 (pinned) |
| PyTorch | 2.4.0 |
| CUDA | 12.4 |
| numpy | < 2.1 |

**GPUs.** Every training archived in this repository ran on a single **NVIDIA
A40 (46 GB)**. The published companion study ran on Google Colab **A100**.

Full install instructions are in [`environment.md`](environment.md).

---

## 2. Reproducing one cell

Each of the trained cells is a single nnU-Net trainer class. To reproduce one:

1. **Install nnU-Net** at the pinned version and set `nnUNet_raw`,
   `nnUNet_preprocessed`, `nnUNet_results` (see `environment.md`).

2. **Copy the trainer file** containing the class you want into nnU-Net's
   trainer directory:

   ```bash
   cp trainers/nnUNetTraine_GN.py \
      "$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/"
   ```

   The batch-size trainer is self-contained. The Batch Norm trainer must be
   installed as `nnUNetTrainer_Advanced.py` if you also want the frozen-alpha
   ablation trainer, which imports `BaseAdvancedTrainer` from that module name.

3. **Plan and preprocess** the dataset (BraTS 2023 adult glioma as dataset 100):

   ```bash
   nnUNetv2_plan_and_preprocess -d 100
   ```

   To reproduce the paper's fold assignment exactly, replace the generated
   `nnUNet_preprocessed/Dataset100_BraTS2023/splits_final.json` with
   [`results/splits_final.json`](results/splits_final.json) from this
   repository before training. The splits are 5-fold `KFold(shuffle=True,
   random_state=12345)` over the sorted case identifiers.

4. **Train all five folds:**

   ```bash
   for FOLD in 0 1 2 3 4; do
     nnUNetv2_train 100 3d_fullres $FOLD -tr <TrainerClass> --npz
   done
   ```

5. **Predict** the validation set with the 5-fold soft-voting ensemble,
   mirroring (test-time augmentation) enabled and `tile_step_size 0.5` — these
   are nnU-Net's defaults; the `--npz` flag above is what makes the softmax
   ensemble possible.

6. **Submit** the resulting segmentations to the **BraTS 2023 Adult Glioma
   continuous-evaluation queue on Synapse** (project `syn51156910`, evaluation
   `9615339`). The per-case CSV the queue returns is the format stored in
   `results/synapse_per_case/`.

### Trainer class for every cell

Class names below are read from the trainer files in `trainers/`, not from the
manuscript. Where a file defines an alias, both names resolve to the same
class and either can be passed to `-tr`.

**In-domain screen — Instance Norm, 12 activations** (`BaseAblationTrainer`)

| # | Activation | Trainer class | File |
|---|---|---|---|
| 1 | LeakyReLU (baseline) | `nnUNetTrainer_500ep_LeakyReLU` | `nnUNetTraine_IN.py` |
| 2 | ReLU | `nnUNetTrainer_500ep_ReLU` | `nnUNetTraine_IN.py` |
| 3 | PReLU | `nnUNetTrainer_500ep_PReLU` | `nnUNetTraine_IN.py` |
| 4 | Swish (SiLU) | `nnUNetTrainer_500ep_Swish` | `nnUNetTraine_IN.py` |
| 5 | TanhExp | `nnUNetTrainer_500ep_TanhExp` | `nnUNetTraine_IN.py` |
| 6 | ELU | `nnUNetTrainer_500ep_ELU` | `nnUNetTraine_IN_remaining.py` |
| 7 | GELU | `nnUNetTrainer_500ep_GELU` | `nnUNetTraine_IN_remaining.py` |
| 8 | Mish | `nnUNetTrainer_500ep_Mish` | `nnUNetTraine_IN_remaining.py` |
| 9 | ELiSH | `nnUNetTrainer_500ep_ELiSH` | `nnUNetTraine_IN_remaining.py` |
| 10 | HardELiSH | `nnUNetTrainer_500ep_HardELiSH` | `nnUNetTraine_IN_remaining.py` |
| 11 | Logish | `nnUNetTrainer_500ep_Logish` | `nnUNetTraine_IN_remaining.py` |
| 12 | Smish | `nnUNetTrainer_500ep_Smish` | `nnUNetTraine_IN_remaining.py` |

The twelve are split across two files on purpose: `ReLU` is defined only in
`nnUNetTraine_IN.py`, because duplicating a class name across two files on the
trainer search path makes nnU-Net's lookup ambiguous.

**Factorial — Batch Norm × 4 activations** (`BaseAdvancedTrainer`,
`nnUNetTrainer_BN.py`)

| # | Activation | Trainer class | Alias |
|---|---|---|---|
| 13 | LeakyReLU | `nnUNetTrainer_Adv_LeakyReLU` | `nnUNetTrainer_500ep_BN_LeakyReLU` |
| 14 | PReLU | `nnUNetTrainer_Adv_PReLU` | `nnUNetTrainer_500ep_BN_PReLU` |
| 15 | Swish (SiLU) | `nnUNetTrainer_Adv_Swish` | `nnUNetTrainer_500ep_BN_Swish` |
| 16 | TanhExp | `nnUNetTrainer_Adv_TanhExp` | `nnUNetTrainer_500ep_BN_TanhExp` |

`nnUNetTrainer_BN.py` also defines `nnUNetTrainer_Adv_ReLU`
(alias `nnUNetTrainer_500ep_BN_ReLU`), which is not one of the four reported
Batch Norm cells.

**Factorial — Group Norm (8 groups) × 4 activations** (`BaseGNTrainer`,
`nnUNetTraine_GN.py`)

| # | Activation | Trainer class | Alias |
|---|---|---|---|
| 17 | LeakyReLU | `nnUNetTrainer_500ep_GN_LeakyReLU` | — |
| 18 | PReLU | `nnUNetTrainer_500ep_GN_PReLU` | `nnUNetTrainer_500ep_GN_PreLU` |
| 19 | Swish (SiLU) | `nnUNetTrainer_500ep_GN_Swish` | — |
| 20 | TanhExp | `nnUNetTrainer_500ep_GN_TanhExp` | — |

The `GN_PreLU` spelling (lower-case `e`) exists because the results folder for
that variant was created with that spelling; nnU-Net resolves a trainer by the
name stored in the checkpoint, so both spellings are exposed as the same class.

**Batch-size experiment — batch 8** (`_BatchSizeBaseTrainer`,
`nnUNetTrainer_batchsize.py`)

| # | Configuration | Trainer class |
|---|---|---|
| 21 | Batch Norm + LeakyReLU, batch 8 | `nnUNetTrainer_500ep_BN_LeakyReLU_bs8` |
| 22 | Group Norm + LeakyReLU, batch 8 | `nnUNetTrainer_500ep_GN_LeakyReLU_bs8` |

`nnUNetTrainer_batchsize.py` additionally defines PReLU and Swish variants at
batch 8, a batch-4 bracket, and a batch-8 full-iteration bracket. Those classes
are provided for completeness and are **not** among the trained cells whose
results appear in `results/synapse_per_case/`.

**Ablation — frozen PReLU slope** (`nnUNetTrainer_PReLU_BN_FrozenAlpha.py`)

| # | Configuration | Trainer class | Alias |
|---|---|---|---|
| 23 | Batch Norm + PReLU, BN affine trainable, α frozen at 0.25 | `nnUNetTrainer_Adv_PReLU_FrozenAlpha` | `nnUNetTrainer_500ep_BN_PReLU_FrozenAlpha` |

That is **23 distinct trained cells**: 12 (Instance Norm screen) + 4 (Batch
Norm) + 4 (Group Norm) + 2 (batch 8) + 1 (ablation). The four Instance Norm
cells of the 3 × 4 factorial are cells 1, 3, 4 and 5 of the screen — they are
not trained twice.

**Not archived here.** The study also includes a frozen-affine Batch Norm
condition, run across all four activations (LeakyReLU, PReLU, Swish, TanhExp),
in which the BatchNorm affine parameters and the PReLU slope were held at their
initial values. Those four cells are described in the paper; their trainer and
per-case metrics are not archived in this repository. The α-only control
retained here (cell 23, `BN_PReLU_frozen.csv`) is the follow-up that separates
the effect of the frozen slope from the effect of the frozen affine. See
"`Adv_` means two different things" below.

---

## 3. Training schedule

Every cell was trained for the **first 500 epochs of nnU-Net's default
1000-epoch polynomial learning-rate schedule**, not for a 500-epoch schedule.
`self.num_epochs = 500` is assigned *after* `super().initialize()` in every
trainer, so the `PolyLRScheduler` is constructed on the 1000-epoch horizon and
training stops at epoch 500 with a learning rate of **≈ 0.0054** rather than
annealing to zero. This ordering is preserved deliberately and identically in
all trainers so that the cells remain comparable; it is a property of the
experiment, not an oversight.

- Batch size **2**, **250** training iterations per epoch (plus 50 validation
  iterations), patch size 128³, AMP, deep supervision, nnU-Net's default
  dataloader and augmentation.
- The batch-8 cells hold samples-per-epoch approximately constant by cutting
  iterations to **62** per epoch (250 × 2 / 8), so a batch-8 epoch costs about
  the same wall-clock time as a batch-2 epoch.

---

## 4. Repository layout

```
README.md
LICENSE                      MIT, copyright 2026 the authors
CITATION.cff
environment.md               exact versions
trainers/                    nnU-Net trainer definitions
benchmark/                   training-cost benchmark outputs and parser
results/synapse_per_case/    per-case official BraTS metrics, one CSV per model
results/splits_final.json    5-fold split used by every cell
notebooks/                   inference / submission / failure-case notebooks
figures/
analysis/                    analysis scripts (see analysis/README.md)
```

### A note on file names

`trainers/nnUNetTraine_IN.py`, `nnUNetTraine_IN_remaining.py` and
`nnUNetTraine_GN.py` carry a misspelling — "Traine", not "Trainer". The names
are kept exactly as used, because renaming them serves no purpose: nnU-Net
resolves trainers by **class** name, and the results folders reference the
class names, not the file names. Only the class names in the tables above
matter when passing `-tr`.

### `Adv_` means two different things — do not conflate them

The prefix `Adv_` appears in two unrelated places, and reading one as the other
would misattribute the results:

- **In the code.** `trainers/nnUNetTrainer_BN.py` defines its classes as
  `nnUNetTrainer_Adv_LeakyReLU`, `nnUNetTrainer_Adv_PReLU`, and so on. This
  file is the **corrected** Batch Norm trainer. The `Adv_` class names were
  kept only so that checkpoints trained under the old names stay loadable;
  `nnUNetTrainer_500ep_BN_*` aliases point at exactly the same classes.
- **In result files.** Files named `Adv_*.csv` exist in the authors' working
  archive and are **not included here**. They come from an earlier version of
  the Batch Norm trainer that performed the IN→BN and activation swaps inside
  `initialize()`, *after* `super().initialize()` had already built the
  optimiser. The BatchNorm affine parameters (γ, β) and the PReLU slope (α)
  were therefore never registered with the optimiser and stayed frozen at
  their initial values for all 500 epochs. Convolutional weights were
  unaffected, which is why those runs still trained and produced plausible
  numbers.

Every Batch Norm result in this repository — the `BN_*.csv` files — comes from
the corrected trainer, with the affine parameters trained normally.

`BN_PReLU_frozen.csv` is a third, distinct condition: a **deliberate** control
in which the BN affine is trainable and only the PReLU slope is frozen at 0.25
(`requires_grad_(False)`), which isolates the effect of the learnable slope
from the effect of the affine. It is not the accidental freeze described above.

---

## 5. Per-case result files

`results/synapse_per_case/` holds one CSV per trained model — 23 files, one for
each cell in the table above — exactly as returned by the Synapse scorer. Every
file has **219 case rows** — the blinded BraTS 2023 validation cases — plus
**5 summary rows** appended by Synapse, labelled `mean`, `std`, `25quantile`,
`median`, `75quantile`. Drop the summary rows before computing per-case
statistics. The 219-row count has been checked in all 23 files.

The first column is unnamed and carries the case identifier
(`BraTS-GLI-XXXXX-000`); the remaining 15 columns are:

| Column family | Meaning |
|---|---|
| `LesionWise_Dice_{ET,TC,WT}` | Lesion-wise Dice for Enhancing Tumour, Tumour Core, Whole Tumour (higher is better) |
| `LesionWise_Hausdorff95_{ET,TC,WT}` | Lesion-wise 95th-percentile Hausdorff distance, mm (lower is better) |
| `Num_TP_{ET,TC,WT}` | True-positive lesions matched |
| `Num_FP_{ET,TC,WT}` | False-positive lesion components |
| `Num_FN_{ET,TC,WT}` | Ground-truth lesions missed |

These are the **official BraTS 2023 lesion-wise metrics**, computed by the
challenge organisers against **sequestered** validation labels. They cannot be
recomputed locally, because the ground truth for these 219 cases is not
released. Per-file details are in
[`results/synapse_per_case/README.md`](results/synapse_per_case/README.md).

---

## 6. Training-cost benchmark

`benchmark/` holds two independent cost measurements.

**`bench12` — real epoch time.** Twelve activations were benchmarked by
training real epochs of the basic (Instance Norm, batch 2) configuration and
reading the per-epoch wall-clock time and peak memory out of the training logs,
with epoch 0 discarded as warm-up. Because runs were executed in concurrent
five-way waves, and the pod's CPUs and network storage are shared across a
wave, raw epoch times from different waves are **not** comparable — the shared
reference activation alone was observed to drift by several percent between
waves under identical settings. Every activation is therefore **normalised
against the LeakyReLU reference measured in its own wave**, and the
wave-normalised table is the one to read. `bench12_sharedref_table.csv` is one
row per activation; `bench12_sharedref_per_epoch.csv` is every epoch of every
wave; `training_cost_table_sharedref.md` is the rendered table, including the
residual-uncertainty check that says how much cross-wave noise survives
normalisation. `parse_bench12.py` is the script that produces all of these from
the training logs.

**`vram_profile_grid.py` — model-only step footprint.** The `vram_grid_*.csv`
files report the VRAM of a forward/backward step of the network alone
(architecture plus activation), without optimiser state or dataloader. This is
the figure to quote when comparing activations' intrinsic memory cost, and it
tracks the real per-epoch peak closely enough to serve as a proxy for it.
`vram_IN_batch2.csv` is the Instance-Norm batch-2 grid, `vram_grid_batch2.csv`
and `vram_grid_batch8.csv` the full norm × activation grids at each batch size.
`bs8_production_epochs.csv` holds the measured epoch times of the production
batch-8 runs, and `gpu.txt` records the GPU the benchmark ran on.

The two measurements answer different questions and will not agree in absolute
terms: `bench12` is the whole training process under production-like
concurrency, `vram_profile_grid.py` is the model in isolation.

---

## 7. Data

**BraTS data are not redistributed in this repository.** No imaging volumes, no
label maps, no prediction NIfTIs or submission archives are included. The
BraTS 2023 adult glioma training and validation data must be obtained from
Synapse under the BraTS 2023 data-use conditions, which you accept directly
with the organisers.

If you use the data, cite:

- B. H. Menze *et al.*, "The Multimodal Brain Tumor Image Segmentation
  Benchmark (BRATS)," *IEEE Transactions on Medical Imaging*, vol. 34, no. 10,
  pp. 1993–2024, 2015.
- S. Bakas *et al.*, "Advancing The Cancer Genome Atlas glioma MRI collections
  with expert segmentation labels and radiomic features," *Scientific Data*,
  vol. 4, 170117, 2017.
- U. Baid *et al.*, "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor
  Segmentation and Radiogenomic Classification," *arXiv:2107.02314*, 2021.

---

## 8. Trained checkpoints

Model weights are not in this repository. Trained checkpoints are **available on
request from the corresponding author; a Zenodo archive is planned.**

---

## 9. Contact

Mohamed A. A. Ahmed — <maaa82@sustech.edu>
