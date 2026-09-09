#!/usr/bin/env python3
"""Figure 12, both parts, in one visual language.

  (a) fig12a.png  Arm A -- BraTS 2020 residual 3D U-Net, single A100-SXM4-40GB.
                  Panel A latency vs accuracy, panel B peak GPU memory vs accuracy.
                  Reproduces the dashboard published in the Arm A paper
                  (Informatics 2026, doi:10.3390/informatics13070118).
  (b) fig12b.png  Arm B -- BraTS 2023 nnU-Net under Instance Normalisation, A40.
                  Panel C measured epoch time vs accuracy, panel D peak training
                  memory vs accuracy.

Every activation keeps the same colour in both parts so the two can be read together.

Data provenance
---------------
Arm A latency / VRAM : timing_summary.csv (median s/epoch and peak VRAM measured on
                       one NVIDIA A100-SXM4-40GB, supplied by the author).
Arm A Dice           : the five-region Dice of the Arm A screen (ET, TC, WT, NCR, ED)
                       as published in the companion Arm A paper; the y-axis is their mean.
                       These are the single-run values of the Arm A results table, not
                       the later three-seed means in Multiseed*/across_seed_summary.
Arm B epoch time     : keep_batchC/production_epoch_times_IN.csv -- median steady-state
                       epoch time read from the archived nnU-Net training logs (fold 0)
                       of the PRODUCTION runs. NOT the concurrent cost benchmark, whose
                       absolute seconds are ~2.5-3x a real fold under contention. ReLU appears in both
                       scheduling batches and is plotted as the control.
Arm B peak memory    : keep_batchC/bench12_sharedref_table.csv (contention-independent)
Arm B Dice           : IN_lesionwise_ALL12.csv (mean lesion-wise Dice over ET/TC/WT,
                       219 blinded validation cases)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------- Arm A Dice
# ET, TC, WT, NCR, ED -- Arm A screen, as published.
ARMA_DICE = {
    "swish":      (0.812, 0.864, 0.871, 0.676, 0.784),
    "tanhexp":    (0.811, 0.864, 0.879, 0.665, 0.791),
    "leaky_relu": (0.809, 0.866, 0.877, 0.659, 0.790),
    "prelu":      (0.794, 0.846, 0.866, 0.664, 0.775),
    "relu":       (0.805, 0.871, 0.871, 0.663, 0.782),
    "elu":        (0.808, 0.867, 0.874, 0.671, 0.786),
    "gelu":       (0.797, 0.854, 0.874, 0.670, 0.787),
    "mish":       (0.805, 0.862, 0.877, 0.661, 0.789),
    "logish":     (0.805, 0.866, 0.877, 0.670, 0.791),
    "smish":      (0.801, 0.860, 0.871, 0.661, 0.785),
    "elish":      (0.800, 0.855, 0.869, 0.660, 0.783),
    "hard_elish": (0.804, 0.865, 0.873, 0.670, 0.784),
}
PRETTY = {"swish": "Swish", "tanhexp": "TanhExp", "leaky_relu": "Leaky ReLU",
          "prelu": "PReLU", "relu": "ReLU", "elu": "ELU", "gelu": "GELU",
          "mish": "Mish", "logish": "Logish", "smish": "Smish",
          "elish": "ELiSH", "hard_elish": "HardELiSH"}
# one colour per activation, shared by both parts
COLOUR = {"leaky_relu": "#1f77b4", "elu": "#2ca02c", "swish": "#d62728",
          "relu": "#000000", "gelu": "#17becf", "hard_elish": "#d43fa0",
          "elish": "#8c564b", "prelu": "#9467bd", "logish": "#bcbd22",
          "mish": "#ff7f0e", "tanhexp": "#ff9896", "smish": "#7f7f7f"}
# marker overrides: the reference activation of each arm, and the Arm A standout
SQUARE = {"a": "relu", "b": "leaky_relu"}       # reference / baseline
STAR = {"a": "swish", "b": None}                # best trade-off (none in Arm B)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.color": "0.90", "grid.linewidth": 0.7})
BAND = "#F7D9D6"


def draw(ax, x, y, arm, labels, title, xlabel, ylabel, zone=None, zone_text=None,
         xerr=None):
    if zone is not None:
        ax.axvspan(*zone, color=BAND, alpha=0.55, zorder=0, lw=0)
        if zone_text:
            ax.annotate(zone_text, xy=(zone[0], 0.02), xycoords=("data", "axes fraction"),
                        xytext=(2, 0), textcoords="offset points",
                        ha="left", va="bottom", fontsize=7, color="#B03A2E", style="italic")
    for k in x.index:
        marker, size = "o", 55
        if k == SQUARE[arm]:
            marker, size = "s", 78
        elif k == STAR[arm]:
            marker, size = "*", 260
        if xerr is not None:
            ax.errorbar(x[k], y[k], xerr=xerr[k], fmt="none", ecolor=COLOUR[k],
                        elinewidth=0.9, capsize=2.0, alpha=0.5, zorder=1)
        ax.scatter(x[k], y[k], marker=marker, s=size, c=COLOUR[k],
                   edgecolors="white", linewidths=0.7, zorder=3)
        dx, dy = labels[k]
        bold = k in (SQUARE[arm], STAR[arm])
        ax.annotate(PRETTY[k], (x[k], y[k]), textcoords="offset points", xytext=(dx, dy),
                    ha="center", va="center", fontsize=7.5, zorder=4,
                    color=COLOUR[k] if not bold else COLOUR[k],
                    fontweight="bold" if bold else "normal",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6,
                                    shrinkA=1, shrinkB=6))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


# ================================================================= part (a)
tim = pd.read_csv(os.path.join(HERE, "timing_summary.csv")).set_index("Activation")
assert set(tim.index) == set(ARMA_DICE), set(tim.index) ^ set(ARMA_DICE)
a = pd.DataFrame({
    "lat": tim.sec_per_epoch_median,
    "vram": tim.peak_vram_gb,
    "dice": pd.Series({k: sum(v) / 5 for k, v in ARMA_DICE.items()}),
})
lat_ref, vram_ref = a.lat["leaky_relu"], a.vram["leaky_relu"]

LAB_A_LAT = {"leaky_relu": (-52, 6), "elu": (-30, 20), "swish": (26, 20), "relu": (-16, -20),
             "gelu": (-40, -14), "hard_elish": (24, 22), "elish": (-14, -22),
             "prelu": (18, -20), "logish": (-30, 20), "mish": (-16, -20),
             "tanhexp": (22, 20), "smish": (24, -16)}
LAB_A_MEM = {"leaky_relu": (-46, 8), "elu": (-26, 20), "swish": (26, 18), "relu": (-16, -20),
             "gelu": (-34, -16), "hard_elish": (26, 20), "elish": (-14, -22),
             "prelu": (20, -20), "logish": (20, 20), "mish": (-18, -20),
             "tanhexp": (-30, 16), "smish": (-30, -8)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.9))
fig.suptitle("Architectural Efficiency Dashboard: BraTS 2020 3D U-Net "
             "(single NVIDIA A100-SXM4-40 GB)", fontsize=12.5, fontweight="bold", y=0.98)
draw(ax1, a.lat, a.dice, "a", LAB_A_LAT, "(A) Latency vs. Accuracy",
     "Training latency on a common GPU (s/epoch, lower is better)",
     "Mean Dice (ET/TC/WT/NCR/ED)",
     zone=(lat_ref * 0.95, lat_ref * 1.05), zone_text="efficient zone ($\\pm$5% of baseline)")
draw(ax2, a.vram, a.dice, "a", LAB_A_MEM, "(B) Memory vs. Accuracy",
     "Peak GPU memory (GB, lower is better)", "Mean Dice (ET/TC/WT/NCR/ED)",
     zone=(vram_ref * 0.95, vram_ref * 1.25), zone_text="efficient zone ($\\leq$1.25$\\times$ baseline)")
ax1.set_xlim(23.2, 37.2)
ax2.set_xlim(4.4, 17.6)
for ax in (ax1, ax2):
    ax.set_ylim(0.7862, 0.8052)
fig.tight_layout(rect=[0, 0.005, 1, 0.945])
fig.savefig(os.path.join(FIGDIR, "Fig12a_ArmA_efficiency.png"), dpi=200, facecolor="white")
plt.close(fig)

# ================================================================= part (b)
bench = pd.read_csv(os.path.join(HERE, "keep_batchC", "bench12_sharedref_table.csv"))
prod = pd.read_csv(os.path.join(HERE, "keep_batchC", "production_epoch_times_IN.csv"))
dice_b = pd.read_csv(os.path.join(HERE, "IN_lesionwise_ALL12.csv"), index_col=0)
dice_b.index = (dice_b.index.str.replace("IN + ", "", regex=False)
                            .str.replace(" (baseline)", "", regex=False).str.strip())
KEY = {"LeakyReLU": "leaky_relu", "ReLU": "relu", "PReLU": "prelu", "ELU": "elu",
       "GELU": "gelu", "Swish": "swish", "Mish": "mish", "ELiSH": "elish",
       "HardELiSH": "hard_elish", "TanhExp": "tanhexp", "Logish": "logish", "Smish": "smish"}
bench = bench.set_index(bench.activation.map(KEY))
dice_b.index = dice_b.index.map(KEY)
# production epoch time: one row per activation, taking the batch each cell was
# actually trained in (the four factorial cells + ReLU ran in apr_may, the eight
# screen-only cells in jul).  ReLU appears in BOTH batches and is kept aside as the
# scheduling control.
prod["key"] = prod.activation.map(KEY)
relu_jul = float(prod[(prod.key == "relu") & (prod.batch == "jul")].median_steady_s.iloc[0])
prod_main = prod[~((prod.key == "relu") & (prod.batch == "jul"))].set_index("key")
b = pd.DataFrame({"epoch": prod_main.median_steady_s, "batch": prod_main.batch,
                  "sd": bench.sd_pct, "mem": bench.peak_alloc_GB, "dice": dice_b.meanDice})
assert len(b) == 12 and b.notna().all().all()
ep_ref, mem_ref = b.epoch["leaky_relu"], b.mem["leaky_relu"]
relu_apr = b.epoch["relu"]

LAB_B_EP = {"leaky_relu": (-30, 12), "elu": (28, -18), "swish": (-30, -10), "relu": (-26, 14),
            "gelu": (-30, -12), "hard_elish": (22, 13), "elish": (-30, 8),
            "prelu": (-28, -12), "logish": (-30, 10), "mish": (26, 12),
            "tanhexp": (26, 8), "smish": (30, 2)}
LAB_B_MEM = {"leaky_relu": (-18, 20), "elu": (-26, -16), "swish": (28, -12), "relu": (-26, 14),
             "gelu": (28, 12), "hard_elish": (-4, 20), "elish": (-6, -20),
             "prelu": (28, -10), "logish": (28, 12), "mish": (-24, 14),
             "tanhexp": (-6, 20), "smish": (28, -12)}

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(11.0, 4.9))
fig.suptitle("Measured Training Cost: BraTS 2023 nnU-Net, Instance Normalisation "
             "(single NVIDIA A40-46 GB)", fontsize=12.5, fontweight="bold", y=0.98)
draw(ax3, b.epoch, b.dice, "b", LAB_B_EP, "(C) Epoch Time vs. Accuracy",
     "Median epoch time from the training logs (s/epoch, lower is better)",
     "Mean lesion-wise Dice (ET/TC/WT)",
     zone=(ep_ref * 0.95, ep_ref * 1.05),
     zone_text="efficient zone ($\\pm$5% of baseline)")
# scheduling control: ReLU re-run in the busier batch, marked as a bar
ax3.axvline(relu_jul, color="0.25", lw=1.2, ls="--", zorder=2)
ax3.annotate("ReLU re-run on a busier\n"
             f"node: {relu_jul:.1f} s (+{100 * (relu_jul / relu_apr - 1):.0f}%,\n"
             "same activation)",
             xy=(relu_jul, 0.985), xycoords=("data", "axes fraction"),
             xytext=(6, 0), textcoords="offset points",
             ha="left", va="top", fontsize=7.5, color="0.25", linespacing=1.4)
draw(ax4, b.mem, b.dice, "b", LAB_B_MEM, "(D) Memory vs. Accuracy",
     "Peak training memory (GB, lower is better)", "Mean lesion-wise Dice (ET/TC/WT)",
     zone=(mem_ref * 0.95, mem_ref * 1.25),
     zone_text="efficient zone ($\\leq$1.25$\\times$ baseline)")
ax3.set_xlim(64, 92)
ax4.set_xlim(3.4, 18.0)
for ax in (ax3, ax4):
    ax.set_ylim(0.7898, 0.8252)
fig.tight_layout(rect=[0, 0.005, 1, 0.945])
fig.savefig(os.path.join(FIGDIR, "Fig12b_ArmB_cost_IN.png"), dpi=200, facecolor="white")
plt.close(fig)

print("wrote Fig12a_ArmA_efficiency.png and Fig12b_ArmB_cost_IN.png")
print("\nArm A (latency s/epoch, VRAM GB, mean 5-region Dice):")
print(a.sort_values("dice", ascending=False).round(4).to_string())
print("\nArm A relative to Leaky ReLU:")
print((a[["lat", "vram"]] / a.loc["leaky_relu", ["lat", "vram"]]).round(3)
      .sort_values("lat").to_string())
print("\nArm B (epoch s, peak GB, mean lesion-wise Dice):")
print(b.sort_values("dice", ascending=False).round(4).to_string())
