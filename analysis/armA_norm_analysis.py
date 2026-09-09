#!/usr/bin/env python3
"""Arm A normalisation x activation factorial (residual 3D U-Net, BraTS 2020).

The published Arm A study held normalisation fixed at Batch Norm and varied only the
activation. The Instance-Norm and Group-Norm cells are unpublished and are what this
paper adds. This script computes the seed-matched, patient-paired comparison.

Design
------
  3 normalisations (IN, BN, GN) x 5 activations (LeakyReLU, PReLU, ReLU, Swish, TanhExp)
  seed 2025 only, so all three normalisations are compared under an identical seed
  73 patients, identical fixed validation split across every cell -> fully paired

Sources
-------
  Normalisation/raw_predictions_norm.csv  -- per-patient IN and GN cells
  all12_per_patient_merged.csv            -- per-patient BN cells (the published arm)

Statistics
----------
  Wilcoxon signed-rank, zeros dropped, normal approximation with tie correction and no
  continuity correction (scipy is unavailable here; the same hand-implementation was used
  for the Arm B batch-size analysis, so the two are consistent). Holm correction is applied
  within each (metric, region) family across the five activations.

Writes armA_norm_stats.csv and prints a summary.
"""
import os
from math import erf, sqrt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2025
ACTS = ["leaky_relu", "prelu", "relu", "swish", "tanhexp"]
REGIONS = ["ET", "TC", "WT", "NCR", "ED"]
PRETTY = {"leaky_relu": "LeakyReLU", "prelu": "PReLU", "relu": "ReLU",
          "swish": "Swish", "tanhexp": "TanhExp"}


def norm_cdf(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def wilcoxon_p(x, y):
    """Two-sided Wilcoxon signed-rank, normal approximation with tie correction."""
    d = [a - b for a, b in zip(x, y) if a - b != 0]
    n = len(d)
    if n < 6:
        return float("nan"), n
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                       # average ranks within ties on |d|
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    mean = n * (n + 1) / 4.0
    tie_term = 0.0
    i = 0
    while i < n:                       # tie correction on the variance
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    var = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var <= 0:
        return float("nan"), n
    z = (w_plus - mean) / sqrt(var)
    return 2.0 * (1.0 - norm_cdf(abs(z))), n


def holm(pairs):
    """pairs: list of (key, p). Returns {key: p_holm}."""
    valid = [(k, p) for k, p in pairs if p == p]
    m = len(valid)
    out, running = {}, 0.0
    for rank, (k, p) in enumerate(sorted(valid, key=lambda t: t[1])):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)     # enforce monotonicity
        out[k] = running
    for k, p in pairs:
        if p != p:
            out[k] = float("nan")
    return out


# ------------------------------------------------------------------ load, seed-matched
raw = pd.read_csv(os.path.join(HERE, "Normalisation", "raw_predictions_norm.csv"))
mrg = pd.read_csv(os.path.join(HERE, "all12_per_patient_merged.csv"))

ing = raw[raw.Seed == SEED].copy()                      # IN and GN cells
bn = mrg[(mrg.Seed == SEED) & (mrg.Activation.isin(ACTS))].copy()
bn["Norm"] = "bn"
cells = pd.concat([ing, bn], ignore_index=True)

patients = sorted(set.intersection(*(
    set(cells[(cells.Norm == n) & (cells.Activation == a)].Patient)
    for n in ("in", "bn", "gn") for a in ACTS)))
assert len(patients) == 73, f"expected 73 shared patients, got {len(patients)}"

def series(norm, act, col):
    s = cells[(cells.Norm == norm) & (cells.Activation == act)].set_index("Patient")[col]
    return [float(s[p]) for p in patients]


# ------------------------------------------------------------------ compare vs IN
rows = []
for metric in ("Dice", "HD95"):
    for reg in REGIONS:
        col = f"{metric}_{reg}"
        for contrast in ("bn", "gn"):
            fam = []
            for act in ACTS:
                ref, alt = series("in", act, col), series(contrast, act, col)
                p, n = wilcoxon_p(alt, ref)
                fam.append(((act, contrast, metric, reg), p))
                rows.append(dict(
                    metric=metric, region=reg, activation=PRETTY[act],
                    comparison=f"{contrast.upper()} vs IN",
                    mean_in=sum(ref) / len(ref), mean_alt=sum(alt) / len(alt),
                    delta=sum(alt) / len(alt) - sum(ref) / len(ref),
                    p_raw=p, n_nonzero=n))
            hp = holm(fam)
            for r in rows[-len(ACTS):]:
                key = ([k for k in hp if PRETTY[k[0]] == r["activation"]
                        and k[1] == contrast and k[2] == metric and k[3] == reg][0])
                r["p_holm"] = hp[key]

df = pd.DataFrame(rows)
df["better"] = df.apply(
    lambda r: (r.delta > 0) if r.metric == "Dice" else (r.delta < 0), axis=1)
df.to_csv(os.path.join(HERE, "armA_norm_stats.csv"), index=False)

# ------------------------------------------------------------------ report
print(f"Arm A normalisation factorial - seed {SEED}, {len(patients)} paired patients\n")
for reg in REGIONS:
    d = df[(df.metric == "Dice") & (df.region == reg)]
    print(f"Dice {reg}:")
    for contrast in ("BN vs IN", "GN vs IN"):
        s = d[d.comparison == contrast]
        sig = (s.p_holm < 0.05).sum()
        worse = (~s.better).sum()
        print(f"   {contrast}: delta {s.delta.min():+.4f}..{s.delta.max():+.4f}  "
              f"| {worse}/5 worse than IN | {sig}/5 significant after Holm")
    print()
print("wrote armA_norm_stats.csv")
