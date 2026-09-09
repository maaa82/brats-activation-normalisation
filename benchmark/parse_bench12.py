#!/usr/bin/env python3
"""
Parse the real-epoch benchmark logs into the training-cost table.

TWO SOURCES, IN ORDER OF PREFERENCE
-----------------------------------
1. Shared-reference wave archive  (/workspace/bench12_shared_ref/<WAVE>/<Act>__training_log_*.txt)
   Written when all twelve are run in five-way waves that each contain a shared
   reference activation. This is the only internally consistent source: raw epoch
   times are NOT comparable between waves (the reference itself has been observed
   to drift ~5% wave to wave under identical settings), so every activation is
   normalised against the reference measured in ITS OWN wave.

2. Live results dir (nnUNetTrainer_bench_IN_<Act>__.../fold_0/training_log_*.txt)
   Newest log per activation. Used for the raw per-activation numbers, and as the
   only source if no wave archive exists. Its `rel.` column is cross-wave and
   therefore unreliable whenever the twelve were not all run in one wave -- it is
   labelled as such.

Also parses the production bs8 runs (nnU-Net's own "Epoch time" lines).

Writes to /workspace/keep:
    training_cost_table.md             wave-normalised table (use this one)
    training_cost_table_sharedref.md   identical copy, kept for existing references
    bench12_sharedref_table.csv        one row per activation, wave-normalised
    bench12_sharedref_per_epoch.csv    every epoch of every wave
    bench12_real_epochs.csv            one row per activation, raw/cross-wave
    bench12_per_epoch.csv              every epoch, raw
    bs8_production_epochs.csv          production batch-8 runs

    python3 /workspace/parse_bench12.py
"""
import glob, os, re, statistics, csv, json
import torch

RES = "/workspace/nnUNet_results/Dataset100_BraTS2023"
OUT = "/workspace/keep"
SHARED_REF_DIR = "/workspace/bench12_shared_ref"
REF_ACT = "LeakyReLU"          # shared reference present in every wave
WAVE_NAMES = ["A", "B", "C"]
os.makedirs(OUT, exist_ok=True)
ACTS = ["LeakyReLU","ReLU","PReLU","ELU","GELU","Swish","Mish","ELiSH","HardELiSH","TanhExp","Logish","Smish"]
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
NPROC = os.cpu_count()

bench_re = re.compile(r"\[bench\] epoch (\d+) \| epoch_time_s ([\d.]+) \| peak_alloc_GB ([\d.]+) \| peak_reserved_GB ([\d.]+)")
et_re    = re.compile(r"Epoch time: ([\d.]+) s")


def newest_log(folder):
    logs = sorted(glob.glob(os.path.join(folder, "training_log_*.txt")), key=os.path.getmtime)
    return logs[-1] if logs else None


def read_bench(path):
    """Return [(epoch, epoch_s, alloc, reserved), ...] from one training log."""
    out = []
    with open(path) as fh:
        for line in fh:
            m = bench_re.search(line)
            if m:
                out.append((int(m[1]), float(m[2]), float(m[3]), float(m[4])))
    return out


def summarise(eps):
    """Drop epoch 0 (warm-up) and summarise the rest. None if nothing usable."""
    use = [e for e in eps if e[0] >= 1]
    if not use:
        return None
    t = [e[1] for e in use]
    return dict(n=len(use), mean=statistics.mean(t),
                sd=statistics.stdev(t) if len(t) > 1 else 0.0,
                median=statistics.median(t),
                alloc=max(e[2] for e in use), resv=max(e[3] for e in use))


# ---------------------------------------------------------------- model-only VRAM
model_only = {}
for cand in (f"{OUT}/vram_IN_batch2.csv", "/workspace/vram_grid_batch2.csv"):
    if os.path.exists(cand):
        with open(cand) as fh:
            for r in csv.DictReader(fh):
                if r.get("norm") == "IN" and r.get("activation") in ACTS:
                    model_only[r["activation"]] = float(r["VRAM_train_GB"])
        if model_only:
            break


# ---------------------------------------------------------------- source 1: wave archive
waves = {}          # (wave, act) -> summary
wave_members = {}   # wave -> [act, ...]
sharedref_per_epoch = []
for w in WAVE_NAMES:
    d = os.path.join(SHARED_REF_DIR, w)
    if not os.path.isdir(d):
        continue
    members = []
    for a in ACTS:
        hits = sorted(glob.glob(os.path.join(d, f"{a}__training_log_*.txt")), key=os.path.getmtime)
        if not hits:
            continue
        eps = read_bench(hits[-1])
        s = summarise(eps)
        if s is None:
            continue
        waves[(w, a)] = s
        members.append(a)
        for e in eps:
            if e[0] >= 1:
                sharedref_per_epoch.append(dict(wave=w, activation=a, epoch=e[0],
                                                epoch_s=e[1], peak_alloc_GB=e[2],
                                                peak_reserved_GB=e[3]))
    if members:
        wave_members[w] = members

# a wave is usable only if it contains the shared reference
usable = [w for w in wave_members if (w, REF_ACT) in waves]
missing_ref = [w for w in wave_members if w not in usable]
covered = sorted({a for (w, a) in waves if w in usable})
have_sharedref = bool(usable) and set(covered) == set(ACTS)

sharedref_rows, ref_by_wave, grand_ref, residuals = [], {}, None, []
if usable:
    ref_by_wave = {w: waves[(w, REF_ACT)]["mean"] for w in usable}
    grand_ref = statistics.mean(ref_by_wave.values())
    for a in ACTS:
        ws = [w for w in usable if (w, a) in waves]
        if not ws:
            continue
        rels = [waves[(w, a)]["mean"] / ref_by_wave[w] for w in ws]
        rel = statistics.mean(rels)
        # an activation measured in >1 wave gives a direct read on how much
        # normalisation residual survives -- the reference itself is trivially 1.000
        if len(ws) > 1 and a != REF_ACT:
            residuals.append((a, ws, rels))
        sharedref_rows.append(dict(
            activation=a, waves="".join(ws), n_epochs_per_wave=waves[(ws[0], a)]["n"],
            rel_to_ref=round(rel, 3), norm_epoch_s=round(rel * grand_ref, 1),
            raw_epoch_s_by_wave=";".join(f"{w}:{waves[(w, a)]['mean']:.1f}" for w in ws),
            sd_pct=round(statistics.mean(100 * waves[(w, a)]["sd"] / waves[(w, a)]["mean"] for w in ws), 1),
            peak_alloc_GB=round(max(waves[(w, a)]["alloc"] for w in ws), 2),
            peak_reserved_GB=round(max(waves[(w, a)]["resv"] for w in ws), 2),
            model_only_GB=model_only.get(a, ""),
            gpu_hours_500ep=round(rel * grand_ref * 500 / 3600, 1)))
    sharedref_rows.sort(key=lambda r: r["rel_to_ref"])


# ---------------------------------------------------------------- source 2: live results dir
rows, per_epoch = [], []
for a in ACTS:
    d = os.path.join(RES, f"nnUNetTrainer_bench_IN_{a}__nnUNetPlans__3d_fullres", "fold_0")
    log = newest_log(d)
    if not log:
        rows.append(dict(activation=a, status="MISSING")); continue
    eps = read_bench(log)
    per_epoch += [dict(activation=a, epoch=e[0], epoch_s=e[1],
                       peak_alloc_GB=e[2], peak_reserved_GB=e[3]) for e in eps]
    s = summarise(eps)
    if s is None:
        rows.append(dict(activation=a, status=f"only {len(eps)} epochs")); continue
    rows.append(dict(activation=a, status="ok", n_epochs_used=s["n"],
        epoch_s_mean=round(s["mean"], 1), epoch_s_sd=round(s["sd"], 1),
        epoch_s_median=round(s["median"], 1),
        peak_alloc_GB=round(s["alloc"], 2), peak_reserved_GB=round(s["resv"], 2),
        gpu_hours_500ep=round(s["mean"] * 500 / 3600, 1),
        final_checkpoint=os.path.exists(os.path.join(d, "checkpoint_final.pth"))))

base = next((r for r in rows if r.get("status") == "ok" and r["activation"] == REF_ACT), None)
for r in rows:
    if r.get("status") == "ok" and base:
        r["rel_time_vs_LeakyReLU"] = round(r["epoch_s_mean"] / base["epoch_s_mean"], 3)
        r["rel_mem_vs_LeakyReLU"]  = round(r["peak_alloc_GB"] / base["peak_alloc_GB"], 3)


# ---------------------------------------------------------------- production bs8
prod = []
for tr in ["nnUNetTrainer_500ep_BN_LeakyReLU_bs8", "nnUNetTrainer_500ep_GN_LeakyReLU_bs8"]:
    for f in range(5):
        d = os.path.join(RES, f"{tr}__nnUNetPlans__3d_fullres", f"fold_{f}")
        ts = []
        for log in sorted(glob.glob(os.path.join(d, "training_log_*.txt"))):
            ts += [float(m[1]) for m in (et_re.search(l) for l in open(log)) if m]
        if ts:
            prod.append(dict(model=tr.replace("nnUNetTrainer_500ep_", ""), fold=f, n_epochs=len(ts),
                             epoch_s_median=round(statistics.median(ts), 1),
                             epoch_s_mean=round(statistics.mean(ts), 1),
                             gpu_hours=round(sum(ts) / 3600, 2)))


# ---------------------------------------------------------------- CSVs
keys = ["activation","status","n_epochs_used","epoch_s_mean","epoch_s_sd","epoch_s_median",
        "rel_time_vs_LeakyReLU","peak_alloc_GB","peak_reserved_GB","rel_mem_vs_LeakyReLU",
        "gpu_hours_500ep","final_checkpoint"]
with open(f"{OUT}/bench12_real_epochs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
with open(f"{OUT}/bench12_per_epoch.csv", "w", newline="") as f:
    if per_epoch:
        w = csv.DictWriter(f, fieldnames=list(per_epoch[0].keys())); w.writeheader(); w.writerows(per_epoch)
with open(f"{OUT}/bs8_production_epochs.csv", "w", newline="") as f:
    if prod:
        w = csv.DictWriter(f, fieldnames=list(prod[0].keys())); w.writeheader(); w.writerows(prod)
if sharedref_rows:
    with open(f"{OUT}/bench12_sharedref_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sharedref_rows[0].keys())); w.writeheader(); w.writerows(sharedref_rows)
if sharedref_per_epoch:
    with open(f"{OUT}/bench12_sharedref_per_epoch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sharedref_per_epoch[0].keys())); w.writeheader(); w.writerows(sharedref_per_epoch)
    json.dump({f"{w_}|{a}": v for (w_, a), v in waves.items()},
              open(f"{OUT}/bench12_sharedref_raw.json", "w"), indent=1)


# ---------------------------------------------------------------- markdown
L = []
L.append("# Training cost per activation, basic nnU-Net (IN, batch 2), fold 0, real epochs\n")
L.append(f"GPU: **{GPU}**, {NPROC} vCPUs. Epoch = 250 training + 50 validation iterations, patch 128³, AMP, "
         "deep supervision, nnU-Net default dataloader/augmentation. Epoch 0 (warm-up) excluded. "
         "Peak memory = `torch.cuda.max_memory_allocated` over the epoch (full process: model, activations, "
         "optimiser state); reserved = allocator high-water mark.\n")

if sharedref_rows:
    L.append("## Wave-normalised (use this table)\n")
    intro = (f"Every activation is normalised against **{REF_ACT}** measured in its own five-way wave, so "
             "concurrent-load differences between waves cancel.")
    if len(ref_by_wave) > 1:
        lo, hi = min(ref_by_wave.values()), max(ref_by_wave.values())
        intro += (" This matters: the shared reference itself measured " +
                  " / ".join(f"{ref_by_wave[w]:.1f}" for w in usable) +
                  f" s in waves {' / '.join(usable)} — a {100*(hi-lo)/lo:.1f}% drift, the same magnitude as the "
                  "entire activation-to-activation spread. Raw epoch times from different waves must not be "
                  "compared directly.")
    else:
        intro += (f" Only one wave ({usable[0]}) is present, so no cross-wave drift could be measured and the "
                  "normalisation is a no-op here.")
    L.append(intro + "\n")
    if not have_sharedref:
        L.append(f"**Warning: incomplete coverage.** Wave archive covers {len(covered)}/{len(ACTS)} activations "
                 f"({', '.join(covered)}). Missing activations appear only in the raw table below.\n")
    if missing_ref:
        L.append(f"**Warning:** wave(s) {', '.join(missing_ref)} lack the reference `{REF_ACT}` and were skipped.\n")
    L.append("| Activation | waves | rel. to ref | norm. epoch (s) | SD% | peak alloc (GB) | model-only (GB) | rel. mem | GPU-h / 500-ep fold |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    ref_alloc = next((r["peak_alloc_GB"] for r in sharedref_rows if r["activation"] == REF_ACT), None)
    for r in sharedref_rows:
        rm = f"{r['peak_alloc_GB']/ref_alloc:.2f}" if ref_alloc else ""
        L.append(f"| {r['activation']} | {r['waves']} | {r['rel_to_ref']} | {r['norm_epoch_s']} | {r['sd_pct']}% | "
                 f"{r['peak_alloc_GB']} | {r['model_only_GB']} | {rm} | {r['gpu_hours_500ep']} |")
    L.append(f"\nReference (`{REF_ACT}`) grand mean = **{grand_ref:.1f} s**.\n")

    rels = [r["rel_to_ref"] for r in sharedref_rows]
    sds  = [r["sd_pct"] for r in sharedref_rows]
    spread = 100 * (max(rels) - min(rels)) / min(rels)
    L.append(f"**Timing.** Full spread across {len(sharedref_rows)} activations is {spread:.1f}%, against a mean "
             f"per-activation epoch-to-epoch SD of {statistics.mean(sds):.1f}%. "
             + ("No activation is distinguishable from any other in training time: at batch 2 with five concurrent "
                "trainings, epoch time is set by dataloading, augmentation and network-filesystem reads, not by the "
                "non-linearity.\n" if spread <= statistics.mean(sds) else
                "The spread exceeds the per-epoch noise, so activation choice may be measurable here — check the "
                "residual below before claiming an ordering.\n"))
    if residuals:
        L.append("**Normalisation residual (validity check).** Activations measured in more than one wave should "
                 "give the same normalised value in each; the discrepancy is the residual uncertainty that survives "
                 "normalisation.\n")
        worst = 0.0
        for a, ws, rl in residuals:
            d = 100 * (max(rl) - min(rl)) / min(rl)
            worst = max(worst, d)
            L.append(f"- `{a}`: " + ", ".join(f"{w}={v:.3f}" for w, v in zip(ws, rl)) + f" → {d:.1f}% residual")
        L.append(f"\nResidual is ~±{worst/2:.1f}%. Differences in the table smaller than this are noise: quote the "
                 "range, not the ordering.\n")
    else:
        L.append("**No cross-wave repeat besides the reference**, so the residual uncertainty of the normalisation "
                 "could not be measured. Put one non-reference activation in two waves to quantify it.\n")
    L.append("**Memory.** Reproducible to ~0.01 GB across independent runs and unaffected by wave composition, so "
             "the memory ranking is reliable even though the timing ranking is not. Differences come from what "
             "autograd must retain: `inplace=True` primitives keep nothing, while activations written as chains of "
             "primitive ops save an intermediate per link over the full patch 128³ × batch 2 activation map.\n")
    if model_only:
        diffs = [r["peak_alloc_GB"] - r["model_only_GB"] for r in sharedref_rows if r["model_only_GB"] != ""]
        if diffs:
            L.append(f"**Model-only column** is `vram_profile_grid.py --grid in-only` (architecture + activation, no "
                     f"optimiser state or dataloader). The real-epoch peak sits {statistics.mean(diffs):+.3f} GB above "
                     f"it, with a spread of only {max(diffs)-min(diffs):.3f} GB across all activations, so the "
                     "model-only figure is a faithful proxy for the real footprint.\n")

    L.append("## Raw per-activation (latest run each; NOT normalised)\n")
    L.append("Kept for traceability. The `rel.` column here compares runs that may come from different waves and is "
             "unreliable whenever the twelve were not all trained in a single wave — prefer the table above.\n")
else:
    L.append(f"**No shared-reference wave archive found at `{SHARED_REF_DIR}`** — falling back to the raw table. "
             "The `rel.` column below is only meaningful if all twelve were trained in one wave.\n")

L.append("| Activation | epochs | epoch time (s) mean ± SD | rel. | peak alloc (GB) | peak reserved (GB) | rel. mem | GPU-h / 500-epoch fold |")
L.append("|---|---|---|---|---|---|---|---|")
for r in sorted([r for r in rows if r.get("status") == "ok"], key=lambda r: r["epoch_s_mean"]):
    L.append(f"| {r['activation']} | {r['n_epochs_used']} | {r['epoch_s_mean']} ± {r['epoch_s_sd']} | "
             f"{r.get('rel_time_vs_LeakyReLU','')} | {r['peak_alloc_GB']} | {r['peak_reserved_GB']} | "
             f"{r.get('rel_mem_vs_LeakyReLU','')} | {r['gpu_hours_500ep']} |")
miss = [r["activation"] for r in rows if r.get("status") != "ok"]
if miss:
    L.append(f"\n**Missing / incomplete:** {', '.join(miss)}")

if prod:
    L.append("\n## Production batch-8 runs (500 real epochs, 5 folds) — same GPU\n")
    L.append("| model | fold | epochs | median epoch (s) | mean epoch (s) | GPU-h |")
    L.append("|---|---|---|---|---|---|")
    for p in prod:
        L.append(f"| {p['model']} | {p['fold']} | {p['n_epochs']} | {p['epoch_s_median']} | {p['epoch_s_mean']} | {p['gpu_hours']} |")

L.append("\n## Caveats\n")
L.append(f"- Five trainings share the pod's {NPROC} vCPUs during every wave, as in production, so CPU-side "
         "augmentation and network-storage reads are included in epoch time. Runs measured under a different "
         "number of concurrent trainings are not comparable — this is what the wave normalisation corrects for.")
L.append("- Memory is the real process footprint at batch 2 (model, activations, optimiser state), not model-only.")
L.append("- fold 0 only; activation cost does not depend on the fold.")
L.append("- `GPU-h / 500-ep fold` extrapolates the epoch time linearly to 500 epochs and excludes the "
         "end-of-training validation prediction, which these benchmark runs skip.")

md = "\n".join(L) + "\n"
open(f"{OUT}/training_cost_table.md", "w").write(md)
open(f"{OUT}/training_cost_table_sharedref.md", "w").write(md)
print(md)
written = ["training_cost_table.md", "training_cost_table_sharedref.md",
           "bench12_real_epochs.csv", "bench12_per_epoch.csv", "bs8_production_epochs.csv"]
if sharedref_rows:
    written += ["bench12_sharedref_table.csv", "bench12_sharedref_per_epoch.csv", "bench12_sharedref_raw.json"]
print(f"wrote {OUT}/: " + ", ".join(written))
