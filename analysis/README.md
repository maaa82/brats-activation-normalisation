# Analysis scripts

Scripts that turn the archived per-case metrics into the statistics, tables and
figures reported in the paper.

| Script | What it does |
|---|---|
| `batchsize_analysis.py` | Paired Wilcoxon signed-rank tests over the 219 blinded Arm B validation cases for the batch-size experiment: BN@8 vs BN@2 and GN@8 vs GN@2 as the primary contrasts, each batch-8 cell against the IN@2 baseline as secondary, with Holm correction within each metric family. |
| `build_tableXV.py` | Collects every Holm-significant comparison across the in-domain screen and the factorial into a single table. |
| `armA_norm_analysis.py` | Paired statistics for the **Arm A** normalisation × activation factorial (residual 3D U-Net, BraTS 2020, 73-case split). Reads `../results/armA_normalisation/`; writes `armA_norm_stats.csv` (100 comparisons). |
| `build_supplement.py` | Builds the full comparison matrix behind the consolidated significance table — every test, significant or not, in four blocks: S1 in-domain screen, S2 normalisation factorial, S3 batch size, S4 Arm A normalisation. Output: `supplementary_full_matrix.csv` (256 rows). |
| `build_supplement_pdf.py` | Renders that matrix as the Supplementary Material PDF. |
| `build_fig12.py` | Two-panel cost-versus-accuracy figure: Arm A (BraTS 2020, A100) and Arm B (BraTS 2023 under Instance Norm, A40). |

## Outputs archived here

- `armA_norm_stats.csv` — 100 paired comparisons against Instance Normalisation
- `supplementary_full_matrix.csv` — 256 rows, the full comparison matrix

## Inputs, and what is missing

Some scripts were written to run in the authors' working directory and read
intermediate CSVs from earlier in that pipeline — for example
`IN_paired_wilcoxon_vs_LeakyReLU.csv`, the per-cell summary table,
`timing_summary.csv` and `IN_lesionwise_ALL12.csv` — which are **not archived
here**. The per-case files they ultimately derive from *are* archived, in
`../results/`. Expect to adjust the input paths at the top of each script, and
in some cases to regenerate an intermediate, before they will run end to end.

`armA_norm_analysis.py` is the exception: it reads only
`../results/armA_normalisation/` and runs against the archived data as-is.

## Dependencies

`pandas` and `numpy`. The Wilcoxon and Student's *t* tests are implemented
directly (normal approximation with tie correction; regularised incomplete beta
for the *t* CDF), so **SciPy is not required**; p-values may differ from SciPy
in the third decimal. `build_fig12.py` and `build_supplement_pdf.py` also need
`matplotlib`.
