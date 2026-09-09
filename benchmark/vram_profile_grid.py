#!/usr/bin/env python3
"""
vram_profile_grid.py
====================
Peak-VRAM profiling across the NORMALISATION x ACTIVATION grid for nnU-Net 3d_fullres
(PlainConvUNet), BraTS 2023. Supersedes vram_profile_runpod.py, which covered
activations at the plans-default normalisation only.

WHY
---
nnU-Net's training logs record epoch time but NOT memory, so speed is recoverable from
the logs and memory is not. Reporting training and inference cost across the
normalisation x activation grid therefore needs a separate memory measurement; the
12-activation in-domain screen adds a second set of configurations that needs one too.

DEFAULT GRID (20 configurations, seconds each)
  IN x all 12 activations      -> the in-domain screen
  BN x the 4 factorial activations
  GN x the 4 factorial activations
Overlapping cells (IN x the 4) are profiled once and reported in both contexts.

WHAT IT MEASURES
----------------
Peak *allocated* VRAM for one training step (forward + backward, AMP autocast, deep
supervision on) and for one inference forward pass, at the real configuration
(patch 128^3, batch 2 by default).

Activation choice does not change parameter count -- every activation except PReLU is
parameterless -- so differences come from intermediate tensors held for the backward
pass. Normalisation choice does change parameters slightly (BN/IN affine vs GroupNorm)
but all three carry 2C learnable parameters here, so counts stay comparable; BN's running
statistics are buffers, not parameters.

It does NOT include dataloader workers, augmentation buffers or optimiser state. Those
are constant across cells and do not affect the ranking, but the figure understates true
process footprint. Report as:
    "peak model VRAM for one training step (batch B, patch 128^3, AMP, deep supervision)"
not as "training memory requirement".

SETUP (fresh pod)
-----------------
    pip install "numpy<2.1" nnunetv2==2.2.1 pandas        # authoritative (uses plans.json)
    # or minimal:
    pip install dynamic-network-architectures pandas

    python vram_profile_grid.py --plans-dir /workspace/nnUNet_preprocessed/Dataset100_BraTS2023

Run on the SAME GPU TYPE used for training, so the memory figures pair with the epoch
times parsed from the training logs. Run twice (--batch 2 and --batch 8) if you are also
doing the batch-size experiment.
"""

import argparse, gc, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = (128, 128, 128)
IN_CH, OUT_CH = 4, 3
DEEP_SUPERVISION = True
GN_GROUPS = 8

ARCH = dict(n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320],
            kernel_sizes=[[3, 3, 3]] * 6, strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
            n_conv_per_stage=[2] * 6, n_conv_per_stage_decoder=[2] * 5, conv_bias=True)

EXPECTED_ACT_MODULES = 44


# ---------------------------------------------------------------- activations
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))
class ELiSH(nn.Module):
    def forward(self, x): return F.elu(x) * torch.sigmoid(x)
class HardELiSH(nn.Module):
    def forward(self, x): return F.elu(x) * F.hardsigmoid(x)
class Logish(nn.Module):
    def forward(self, x): return x * torch.log(1 + torch.sigmoid(x))
class Smish(nn.Module):
    def forward(self, x): return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))
class TanhExp(nn.Module):
    def forward(self, x): return x * torch.tanh(torch.exp(torch.clamp(x, max=20)))

ACT_SOURCE = "inline definitions"
try:
    from nnunetv2.training.nnUNetTrainer.custom_brats_activations_remaining import (  # noqa
        Mish, ELiSH, HardELiSH, Logish, Smish)
    ACT_SOURCE = "custom_brats_activations_remaining.py (project module)"
except Exception:
    pass

ACTS = {
    "LeakyReLU": (nn.LeakyReLU, {"inplace": True}),
    "ReLU":      (nn.ReLU,      {"inplace": True}),
    "PReLU":     (nn.PReLU,     {}),
    "Swish":     (nn.SiLU,      {"inplace": True}),
    "TanhExp":   (TanhExp,      {}),
    "ELU":       (nn.ELU,       {"inplace": True}),
    "GELU":      (nn.GELU,      {}),
    "Mish":      (Mish,         {}),
    "ELiSH":     (ELiSH,        {}),
    "HardELiSH": (HardELiSH,    {}),
    "Logish":    (Logish,       {}),
    "Smish":     (Smish,        {}),
}

FACTORIAL_ACTS = ["LeakyReLU", "PReLU", "Swish", "TanhExp"]   # the 3 x 4 factorial

NONLIN_NAMES = {"ReLU", "LeakyReLU", "ELU", "GELU", "SiLU", "PReLU",
                "Mish", "ELiSH", "HardELiSH", "Logish", "Smish", "TanhExp"}
NORM_NAMES = {"InstanceNorm3d", "BatchNorm3d", "GroupNorm"}


# ---------------------------------------------------------------- swaps
def swap_act(module, cls, kwargs):
    n = 0
    for name, child in module.named_children():
        if type(child).__name__ in NONLIN_NAMES:
            setattr(module, name, cls(**kwargs)); n += 1
        else:
            n += swap_act(child, cls, kwargs)
    return n


def swap_norm(module, kind):
    """kind in {'in','bn','gn'} — replace every norm layer, preserving channel count."""
    n = 0
    for name, child in module.named_children():
        t = type(child).__name__
        if t in NORM_NAMES:
            C = getattr(child, "num_features", None)
            if C is None:
                C = getattr(child, "num_channels", None)
            if C is None:
                raise RuntimeError(f"cannot read channel count from {t}")
            if kind == "bn":
                new = nn.BatchNorm3d(C)
            elif kind == "gn":
                if C % GN_GROUPS:
                    raise RuntimeError(f"{C} channels not divisible by {GN_GROUPS} groups")
                new = nn.GroupNorm(GN_GROUPS, C)
            else:
                new = nn.InstanceNorm3d(C, affine=True)
            setattr(module, name, new); n += 1
        else:
            n += swap_norm(child, kind)
    return n


# ---------------------------------------------------------------- builders
def build_from_spec():
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    return PlainConvUNet(
        input_channels=IN_CH, n_stages=ARCH["n_stages"],
        features_per_stage=ARCH["features_per_stage"], conv_op=nn.Conv3d,
        kernel_sizes=ARCH["kernel_sizes"], strides=ARCH["strides"],
        n_conv_per_stage=ARCH["n_conv_per_stage"], num_classes=OUT_CH,
        n_conv_per_stage_decoder=ARCH["n_conv_per_stage_decoder"],
        conv_bias=ARCH["conv_bias"], norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        deep_supervision=DEEP_SUPERVISION)


class PlansBuilder:
    def __init__(self, plans_dir):
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from batchgenerators.utilities.file_and_folder_operations import load_json, join
        pf = next((os.path.join(plans_dir, c) for c in ("plans.json", "nnUNetPlans.json")
                   if os.path.exists(os.path.join(plans_dir, c))), None)
        if pf is None:
            raise FileNotFoundError(f"no plans.json / nnUNetPlans.json in {plans_dir}")
        dj = os.path.join(plans_dir, "dataset.json")
        if not os.path.exists(dj):
            raise FileNotFoundError(f"no dataset.json in {plans_dir}")
        self.plans = PlansManager(load_json(pf)); self.dj = load_json(dj)
        self.cm = self.plans.get_configuration("3d_fullres")
        self.n_in = len(self.dj.get("channel_names", self.dj.get("modality", {})))
        self.src = pf

    def __call__(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        return nnUNetTrainer.build_network_architecture(
            self.plans, self.dj, self.cm, self.n_in,
            enable_deep_supervision=DEEP_SUPERVISION)


# ---------------------------------------------------------------- measurement
def measure(net, n_in, batch):
    net = net.cuda()
    x = torch.randn(batch, n_in, *PATCH, device="cuda")

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    net.eval()
    with torch.no_grad(), torch.autocast("cuda"):
        net(x)
    infer = torch.cuda.max_memory_allocated() / 1e9

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    net.train()
    with torch.autocast("cuda"):
        out = net(x)
        loss = (sum(o.float().mean() for o in out)
                if isinstance(out, (list, tuple)) else out.float().mean())
    loss.backward()
    train = torch.cuda.max_memory_allocated() / 1e9

    params = sum(p.numel() for p in net.parameters())
    del x, out, loss
    return infer, train, params


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans-dir", default="/workspace/nnUNet_preprocessed/Dataset100_BraTS2023")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", choices=["default", "full", "in-only"], default="default",
                    help="default = IN x12 + BN x4 + GN x4 ; full = 3 x 12 ; in-only = IN x12")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA device.")

    out = args.out or f"vram_grid_batch{args.batch}.csv"

    print("=" * 84)
    print("nnU-Net 3d_fullres — peak VRAM across normalisation x activation")
    print("=" * 84)
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"total VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"torch            : {torch.__version__}")
    print(f"patch / batch    : {PATCH} / {args.batch}")
    print(f"deep supervision : {DEEP_SUPERVISION}")
    print(f"activation source: {ACT_SOURCE}")

    builder, n_in, mode = None, IN_CH, None
    try:
        pb = PlansBuilder(args.plans_dir)
        builder, n_in, mode = pb, pb.n_in, f"plans ({pb.src})"
    except Exception as e:
        try:
            import dynamic_network_architectures  # noqa
            builder, mode = build_from_spec, "spec (hand-written from the training log)"
            print(f"plans            : unavailable ({e})")
        except Exception:
            sys.exit("ERROR: need nnunetv2+plans.json, or dynamic-network-architectures.")
    print(f"build mode       : {mode}\n")

    if args.grid == "in-only":
        cells = [("IN", a) for a in ACTS]
    elif args.grid == "full":
        cells = [(n, a) for n in ("IN", "BN", "GN") for a in ACTS]
    else:
        cells = ([("IN", a) for a in ACTS]
                 + [(n, a) for n in ("BN", "GN") for a in FACTORIAL_ACTS])

    print(f"{len(cells)} configurations\n")
    print(f"  {'norm':4s} {'activation':11s} {'nAct':>4s} {'nNorm':>5s} {'params(M)':>10s} "
          f"{'infer(GB)':>10s} {'train(GB)':>10s}")
    print("  " + "-" * 62)

    rows, warn = [], []
    for norm, act in cells:
        cls, kw = ACTS[act]
        try:
            net = builder()
            n_norm = swap_norm(net, norm.lower())
            n_act = swap_act(net, cls, kw)
            infer, train, params = measure(net, n_in, args.batch)
            rows.append({"norm": norm, "activation": act,
                         "n_act_modules": n_act, "n_norm_modules": n_norm,
                         "params_M": round(params / 1e6, 3),
                         "VRAM_infer_GB": round(infer, 3),
                         "VRAM_train_GB": round(train, 3)})
            flag = "" if n_act == EXPECTED_ACT_MODULES else f"  <- expected {EXPECTED_ACT_MODULES}"
            if flag: warn.append((norm, act, n_act))
            print(f"  {norm:4s} {act:11s} {n_act:4d} {n_norm:5d} {params/1e6:10.3f} "
                  f"{infer:10.3f} {train:10.3f}{flag}")
            del net
        except Exception as e:
            print(f"  {norm:4s} {act:11s} FAILED: {type(e).__name__}: {e}")
        gc.collect(); torch.cuda.empty_cache()

    if not rows:
        sys.exit("nothing measured")

    import pandas as pd
    df = pd.DataFrame(rows)
    df["GPU"] = torch.cuda.get_device_name(0)
    df["batch"] = args.batch
    df["patch"] = "x".join(map(str, PATCH))
    df["deep_supervision"] = DEEP_SUPERVISION

    print("\n" + "=" * 84)
    print("TRAIN VRAM (GB) — norm x activation")
    print(df.pivot_table(index="activation", columns="norm",
                         values="VRAM_train_GB").to_string())
    base = df[(df.norm == "IN") & (df.activation == "LeakyReLU")]
    if len(base):
        b = base.VRAM_train_GB.iloc[0]
        df["vs_IN_LeakyReLU_%"] = ((df.VRAM_train_GB / b - 1) * 100).round(2)
        print(f"\nreference IN+LeakyReLU = {b:.3f} GB")
        top = df.sort_values("VRAM_train_GB", ascending=False).head(5)
        print("highest 5:")
        for _, r in top.iterrows():
            print(f"   {r.norm:3s} {r.activation:11s} {r.VRAM_train_GB:6.3f} GB "
                  f"({r['vs_IN_LeakyReLU_%']:+.1f}%)")

    df.to_csv(out, index=False)
    print(f"\nsaved -> {os.path.abspath(out)}")
    if warn:
        print(f"\nWARNING: unexpected activation-module count in {warn} — those rows are "
              f"not comparable; investigate before reporting.")
    print("\nReport as: peak MODEL VRAM for one training step "
          f"(batch {args.batch}, patch {'x'.join(map(str,PATCH))}, AMP, "
          f"deep supervision={DEEP_SUPERVISION}). Excludes dataloader, augmentation "
          "and optimiser state.")


if __name__ == "__main__":
    main()
