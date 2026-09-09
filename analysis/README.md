# Analysis scripts

Scripts that consume `../results/synapse_per_case/` and produce the summary
statistics reported in the paper.

| Script | What it does |
|---|---|
| `batchsize_analysis.py` | Paired Wilcoxon signed-rank tests over the 219 blinded validation cases for the batch-size experiment: BN@8 vs BN@2 and GN@8 vs GN@2 as the primary contrasts, each batch-8 cell against the IN@2 baseline as secondary, with Holm correction applied within each metric family. Implements the signed-rank test directly (ranking, tie correction, normal approximation) so it has no SciPy dependency. |
| `build_tableXV.py` | Collects every Holm-significant comparison across the in-domain screen and the factorial into a single table. |

## Inputs

These scripts were written to run in the authors' working directory and read
intermediate CSVs produced earlier in that pipeline — for example
`IN_paired_wilcoxon_vs_LeakyReLU.csv` and the per-cell summary table — which
are not archived here. The per-case metric files they ultimately derive from
**are** archived, in `../results/synapse_per_case/`. Expect to adjust the input
paths at the top of each script before running them.

## Dependencies

`pandas` and `numpy` only.
