#!/usr/bin/env python3
"""Full comparison matrix for the Supplementary Material.

The paper reports a consolidated significance table and refers to the full matrix behind
it, including the non-significant tests and paired t-test p-values as a secondary marker.
This script produces that matrix.

Contents
--------
  S1  In-domain screen        11 activations vs IN + LeakyReLU, 3 regions x 2 metrics
  S2  Normalisation factorial 11 cells       vs IN + LeakyReLU, 3 regions x 2 metrics
  S3  Batch-size experiment   5 paired contrasts,               3 regions x 2 metrics
  S4  Arm A normalisation     from armA_norm_stats.csv (residual U-Net, BraTS 2020)

Every row carries the mean paired difference, the Wilcoxon signed-rank p (raw and
Holm-corrected) and the paired Student's t-test p as a secondary marker.

Holm families:
  screen     -- within each metric family (33 tests: 11 activations x 3 regions)
  factorial  -- across the three subregions of each model and metric
  batch size -- across the three subregions of each contrast and metric

scipy is not available here, so the Wilcoxon normal approximation (with tie correction, no
continuity correction) and the Student's t CDF (regularised incomplete beta, continued
fraction) are implemented directly. p-values may differ from scipy in the third decimal.

Writes supplementary_full_matrix.csv.
"""
import os
from math import erf, exp, lgamma, sqrt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SYN = os.path.join(HERE, "Synapse")
REGIONS = ["ET", "TC", "WT"]
METRICS = {"Dice": "LesionWise_Dice_{}", "HD95": "LesionWise_Hausdorff95_{}"}

SCREEN = {  # in-domain screen, all under Instance Normalisation
    "LeakyReLU": "LeakyReLU_IN.csv", "ReLU": "ReLU_IN.csv", "PReLU": "PReLU_IN.csv",
    "ELU": "ELU_IN.csv", "GELU": "GELU_IN.csv", "Swish": "Swish_IN.csv",
    "Mish": "Mish_IN.csv", "ELiSH": "Elish_IN.csv", "HardELiSH": "HardElish_IN.csv",
    "TanhExp": "TanhExp_IN.csv", "Logish": "Logish_IN.csv", "Smish": "Smish_IN.csv",
}
FACTORIAL = {
    "IN + LeakyReLU": "LeakyReLU_IN.csv", "IN + PReLU": "PReLU_IN.csv",
    "IN + Swish": "Swish_IN.csv", "IN + TanhExp": "TanhExp_IN.csv",
    "BN + LeakyReLU": "BN_LeakyReLU.csv", "BN + PReLU": "BN_PReLU.csv",
    "BN + Swish": "BN_Swish.csv", "BN + TanhExp": "BN_TanhExp.csv",
    "GN + LeakyReLU": "GN_LeakyReLU.csv", "GN + PReLU": "GN_PReLU.csv",
    "GN + Swish": "GN_Swish.csv", "GN + TanhExp": "GN_TanhExp.csv",
}
BATCH = {"BN + LeakyReLU, batch 8": "BN_LeakyReLU_bs8.csv",
         "GN + LeakyReLU, batch 8": "GN_LeakyReLU_bs8.csv"}


# ----------------------------------------------------------------- distributions
def norm_cdf(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def betacf(a, b, x, itmax=300, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30: d = 1e-30
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30: d = 1e-30
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = lgamma(a + b) - lgamma(a) - lgamma(b) + a * __import__("math").log(x) \
        + b * __import__("math").log(1.0 - x)
    front = exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """Two-sided p for Student's t."""
    if df <= 0 or t != t:
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def paired_t_p(x, y):
    d = [a - b for a, b in zip(x, y)]
    n = len(d)
    if n < 3:
        return float("nan")
    m = sum(d) / n
    var = sum((v - m) ** 2 for v in d) / (n - 1)
    if var <= 0:
        return 1.0 if m == 0 else 0.0
    t = m / sqrt(var / n)
    return t_sf_two_sided(t, n - 1)


def wilcoxon_p(x, y):
    d = [a - b for a, b in zip(x, y) if a - b != 0]
    n = len(d)
    if n < 6:
        return float("nan"), n
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    mean = n * (n + 1) / 4.0
    tie = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        t = j - i + 1
        if t > 1:
            tie += t ** 3 - t
        i = j + 1
    var = (n * (n + 1) * (2 * n + 1) - tie / 2.0) / 24.0
    if var <= 0:
        return float("nan"), n
    return 2.0 * (1.0 - norm_cdf(abs((w_plus - mean) / sqrt(var)))), n


def holm(items):
    """items: list of (key, p) -> {key: p_holm}"""
    valid = [(k, p) for k, p in items if p == p]
    m, out, run = len(valid), {}, 0.0
    for rank, (k, p) in enumerate(sorted(valid, key=lambda t: t[1])):
        run = max(run, min(1.0, (m - rank) * p))
        out[k] = run
    for k, p in items:
        if p != p:
            out[k] = float("nan")
    return out


# ----------------------------------------------------------------- load + verify
def load(fn):
    d = pd.read_csv(os.path.join(SYN, fn), index_col=0)
    d = d[d.index.astype(str).str.startswith("BraTS")]
    assert len(d) == 219, f"{fn}: expected 219 cases, got {len(d)}"
    return d

cells = {k: load(v) for k, v in {**SCREEN, **FACTORIAL, **BATCH}.items()}
cases = sorted(cells["IN + LeakyReLU"].index)
for k, d in cells.items():
    assert sorted(d.index) == cases, f"{k}: case list differs"
print(f"loaded {len(cells)} distinct cells, {len(cases)} shared cases\n")

# integrity gate: the factorial must reproduce the reported factorial values
CHECK = {"IN + LeakyReLU": (0.7708, 46.12), "BN + Swish": (0.7267, 65.07),
         "GN + PReLU": (0.7664, 48.30), "BN + TanhExp": (0.7305, 65.52)}
for k, (dice, hd) in CHECK.items():
    got = (cells[k]["LesionWise_Dice_ET"].mean(), cells[k]["LesionWise_Hausdorff95_ET"].mean())
    assert abs(got[0] - dice) < 5e-4 and abs(got[1] - hd) < 5e-2, f"{k} mismatch: {got}"
print("integrity gate passed: factorial cells reproduce Tables 11 and 12\n")


def col(cell, metric, reg):
    return [float(v) for v in cells[cell].loc[cases, METRICS[metric].format(reg)]]


rows = []


def add(block, comparison, a, b, metric, reg, fam):
    x, y = col(a, metric, reg), col(b, metric, reg)
    p, n = wilcoxon_p(x, y)
    delta = sum(x) / len(x) - sum(y) / len(y)
    rows.append(dict(block=block, comparison=comparison, region=reg, metric=metric,
                     delta=delta, p_wilcoxon_raw=p, p_ttest=paired_t_p(x, y),
                     n_nonzero=n, _fam=fam, _key=(block, comparison, region_key := reg, metric)))


# S1 -- in-domain screen, Holm within each metric family
for metric in METRICS:
    for act in [a for a in SCREEN if a != "LeakyReLU"]:
        for reg in REGIONS:
            add("S1 in-domain screen", f"IN + {act} vs IN + LeakyReLU",
                act, "LeakyReLU", metric, reg, f"S1|{metric}")

# S2 -- factorial, Holm across the three subregions of each model and metric
for cell in [c for c in FACTORIAL if c != "IN + LeakyReLU"]:
    for metric in METRICS:
        for reg in REGIONS:
            add("S2 factorial", f"{cell} vs IN + LeakyReLU",
                cell, "IN + LeakyReLU", metric, reg, f"S2|{cell}|{metric}")

# S3 -- batch size, Holm across the three subregions of each contrast and metric
BATCH_CONTRASTS = [
    ("BN + LeakyReLU, batch 8 vs batch 2", "BN + LeakyReLU, batch 8", "BN + LeakyReLU"),
    ("GN + LeakyReLU, batch 8 vs batch 2", "GN + LeakyReLU, batch 8", "GN + LeakyReLU"),
    ("BN + LeakyReLU, batch 8 vs IN + LeakyReLU", "BN + LeakyReLU, batch 8", "IN + LeakyReLU"),
    ("BN + LeakyReLU, batch 8 vs GN + LeakyReLU, batch 8", "BN + LeakyReLU, batch 8", "GN + LeakyReLU, batch 8"),
]
for name, a, b in BATCH_CONTRASTS:
    for metric in METRICS:
        for reg in REGIONS:
            add("S3 batch size", name, a, b, metric, reg, f"S3|{name}|{metric}")

# apply Holm within each declared family
fams = {}
for r in rows:
    fams.setdefault(r["_fam"], []).append((r["_key"], r["p_wilcoxon_raw"]))
adj = {}
for f, items in fams.items():
    adj.update(holm(items))
for r in rows:
    r["p_wilcoxon_holm"] = adj[r["_key"]]
    r["direction"] = ("better" if (r["delta"] > 0 if r["metric"] == "Dice" else r["delta"] < 0)
                      else "worse")
    r["significant"] = "yes" if (r["p_wilcoxon_holm"] == r["p_wilcoxon_holm"]
                                 and r["p_wilcoxon_holm"] < 0.05) else "no"
    del r["_fam"], r["_key"]

df = pd.DataFrame(rows)

# S4 -- Arm A normalisation factorial, already computed against its own reference
arm_a = os.path.join(HERE, "armA_norm_stats.csv")
if os.path.exists(arm_a):
    a4 = pd.read_csv(arm_a)
    a4 = pd.DataFrame(dict(
        block="S4 Arm A normalisation (residual U-Net, BraTS 2020)",
        comparison=a4.comparison + ", " + a4.activation, region=a4.region, metric=a4.metric,
        delta=a4.delta, p_wilcoxon_raw=a4.p_raw, p_ttest=float("nan"),
        n_nonzero=a4.n_nonzero, p_wilcoxon_holm=a4.p_holm,
        direction=a4.better.map({True: "better", False: "worse"}),
        significant=(a4.p_holm < 0.05).map({True: "yes", False: "no"})))
    df = pd.concat([df, a4], ignore_index=True)

df = df[["block", "comparison", "region", "metric", "delta", "p_wilcoxon_raw",
         "p_wilcoxon_holm", "p_ttest", "direction", "significant", "n_nonzero"]]
df.to_csv(os.path.join(HERE, "supplementary_full_matrix.csv"), index=False)

print(df.groupby("block").agg(tests=("delta", "size"),
                              significant=("significant", lambda s: (s == "yes").sum()),
                              improvements=("direction", lambda s: 0)).to_string())
print()
for b, g in df.groupby("block"):
    sig = g[g.significant == "yes"]
    print(f"{b}: {len(g)} tests, {len(sig)} significant after Holm, "
          f"{(sig.direction == 'better').sum()} of them favourable")
print("\nwrote supplementary_full_matrix.csv")
