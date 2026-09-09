import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# ---------------------------------------------------------------------------
# Batch-size experiment: Batch Norm vs Group Norm at batch 8. SELF-CONTAINED.
#
# Everything needed is defined in this one file: the activations, the
# normalisation swaps, the base trainer and the batch-size overrides. It
# imports nothing from the other trainer files, so only this file has to be
# present on the training machine.
#
# NO CLASS NAME HERE COLLIDES with the other trainer files. Duplicating a class
# name across two files in the trainer search path makes nnU-Net's lookup
# ambiguous, so the batch-2 cells (nnUNetTrainer_500ep_ELU etc.) are
# deliberately NOT redefined.
#
# WHAT THE EXPERIMENT DOES
#   Batch Norm is the condition under test. Group Norm is the control: Group
#   Norm computes its statistics within channel groups of a SINGLE sample, so
#   it is batch-independent by construction.
#       BN improves, GN does not  -> the weakness was the small-batch statistics
#       neither improves          -> not a batch-size effect
#       both improve similarly    -> the changed regime, not the normalisation
#
#   Gradient accumulation would NOT answer this: BatchNorm computes statistics
#   over the actual forward-pass batch, so accumulating over steps of batch 2
#   still leaves BN seeing batch 2. Only a larger REAL batch changes them.
#
# THREE THINGS DELIBERATELY MATCHED TO THE BATCH-2 CELLS
#   1. num_epochs is set AFTER super().initialize(), exactly as in the other
#      trainers. nnU-Net therefore builds the poly LR scheduler on its
#      1000-epoch default and training stops at epoch 500 with lr ~0.0054
#      rather than annealing to ~0. Verified against the logs (epoch 499 ->
#      0.00537, which is the 1000-epoch curve). Reproduced on purpose: every
#      other cell behaves this way and the comparison depends on it. Describe
#      it as "the first 500 epochs of nnU-Net's default 1000-epoch polynomial
#      schedule", not as "500 epochs".
#   2. The activation swap replaces the nnU-Net default nonlinearity only,
#      as in the Instance Norm trainers.
#   3. Activation definitions are byte-identical to those used in the
#      companion experiments. Do not rewrite them (e.g. log1p for log(1+.));
#      comparability depends on both sets using the same functions.
#
# ONE DELIBERATE IMPROVEMENT
#   build_network_architecture is a @classmethod here, not an instance method.
#   nnU-Net calls it on the INSTANCE during training but on the CLASS during
#   inference (predict_from_raw_data). The instance-method form works for
#   training and fails at inference with
#       TypeError: ... missing 1 required positional argument
#   A classmethod binds correctly in both cases. The network produced is
#   identical.
#
# VRAM at patch 128^3: batch 2 ~12-16 GB | batch 4 ~24-32 GB | batch 8 ~48-64 GB.
# Run vram_profile_grid.py --batch 8 before launching.
# ---------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Activations (identical to those used in the companion experiments)
# --------------------------------------------------------------------------
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ELiSH(nn.Module):
    def forward(self, x):
        return F.elu(x) * torch.sigmoid(x)


class HardELiSH(nn.Module):
    def forward(self, x):
        return F.elu(x) * F.hardsigmoid(x)


class Logish(nn.Module):
    def forward(self, x):
        return x * torch.log(1 + torch.sigmoid(x))


class Smish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))


class TanhExp(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(torch.clamp(x, max=20)))


# --------------------------------------------------------------------------
# Base trainer: normalisation swap + activation swap + batch-size override
# --------------------------------------------------------------------------
BASE_BATCH = 2
BASE_TRAIN_ITERS = 250
BASE_VAL_ITERS = 50
NUM_EPOCHS = 500


class _BatchSizeBaseTrainer(nnUNetTrainer):
    """Not used directly. Subclasses set NORM, ACTIVATION, BATCH_SIZE."""

    NORM = "in"                 # "in" | "bn" | "gn"
    ACTIVATION = None           # None = keep the nnU-Net default LeakyReLU
    GN_GROUPS = 8               # matches BaseGNTrainer
    BN_KWARGS = {}              # PyTorch defaults: eps 1e-5, momentum 0.1, affine True
    BATCH_SIZE = None
    HOLD_SAMPLES_PER_EPOCH = True

    # ---- construction ----------------------------------------------------
    # Both swaps happen inside build_network_architecture, i.e. BEFORE the
    # optimiser is created in super().initialize(). This is the corrected
    # ordering: the earlier BN trainer swapped after the optimiser existed, so
    # BN's affine parameters were never registered and never trained.
    @staticmethod
    def _swap_norm(network, kind, gn_groups):
        n = 0

        def replace(module):
            nonlocal n
            for name, child in module.named_children():
                t = type(child).__name__
                if t in ("InstanceNorm3d", "BatchNorm3d", "GroupNorm"):
                    C = getattr(child, "num_features", None)
                    if C is None:
                        C = getattr(child, "num_channels", None)
                    if C is None:
                        raise RuntimeError(f"cannot read channel count from {t}")
                    if kind == "bn":
                        new = nn.BatchNorm3d(C, **_BatchSizeBaseTrainer.BN_KWARGS)
                    elif kind == "gn":
                        if C % gn_groups:
                            raise RuntimeError(
                                f"{C} channels not divisible by {gn_groups} groups")
                        new = nn.GroupNorm(gn_groups, C)
                    else:
                        new = nn.InstanceNorm3d(C, eps=1e-5, affine=True)
                    setattr(module, name, new)
                    n += 1
                else:
                    replace(child)

        replace(network)
        return network, n

    @staticmethod
    def _swap_act(network, new_activation_class):
        n = 0

        def replace(module):
            nonlocal n
            for name, child in module.named_children():
                if isinstance(child, nn.LeakyReLU):
                    setattr(module, name, new_activation_class())
                    n += 1
                else:
                    replace(child)

        replace(network)
        return network, n

    @classmethod
    def build_network_architecture(cls, plans_manager, dataset_json,
                                   configuration_manager, num_input_channels,
                                   enable_deep_supervision=True):
        network = nnUNetTrainer.build_network_architecture(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, enable_deep_supervision)

        network, _ = cls._swap_norm(network, cls.NORM, cls.GN_GROUPS)
        if cls.ACTIVATION is not None:
            network, _ = cls._swap_act(network, cls.ACTIVATION)
        return network

    # ---- training configuration ------------------------------------------
    def initialize(self):
        super().initialize()

        # matched to BaseAblationTrainer -- see header note 1
        self.num_epochs = NUM_EPOCHS
        self.num_processes_training = 4

        bs = self.BATCH_SIZE
        self.configuration_manager.configuration["batch_size"] = bs

        if self.HOLD_SAMPLES_PER_EPOCH:
            scale = BASE_BATCH / bs
            self.num_iterations_per_epoch = max(1, round(BASE_TRAIN_ITERS * scale))
            self.num_val_iterations_per_epoch = max(1, round(BASE_VAL_ITERS * scale))

        act = getattr(self.ACTIVATION, "__name__", "LeakyReLU (default)")
        self.print_to_log_file(
            f"[batch-size experiment] norm={self.NORM.upper()} activation={act} "
            f"batch={bs} (reference {BASE_BATCH}) | "
            f"train_iters/epoch={self.num_iterations_per_epoch} | "
            f"val_iters/epoch={self.num_val_iterations_per_epoch} | "
            f"samples/epoch={self.num_iterations_per_epoch * bs} "
            f"(reference {BASE_TRAIN_ITERS * BASE_BATCH}) | epochs={self.num_epochs}")


# ===========================================================================
# BATCH C -- batch 8, iterations cut to 62 so samples/epoch stays ~500.
#            ~12 h/fold, same as the trained batch-2 cells.
# ===========================================================================

# --- minimal pair: the 10 folds that answer the batch-size question -------
class nnUNetTrainer_500ep_BN_LeakyReLU_bs8(_BatchSizeBaseTrainer):
    NORM = "bn"
    BATCH_SIZE = 8


class nnUNetTrainer_500ep_GN_LeakyReLU_bs8(_BatchSizeBaseTrainer):
    NORM = "gn"
    BATCH_SIZE = 8


# --- BATCH D: does the activation x BN interaction survive a larger batch? -
# PReLU and Swish are the two activations whose behaviour under batch-level
# statistics the batch-2 results single out.
class nnUNetTrainer_500ep_BN_PReLU_bs8(_BatchSizeBaseTrainer):
    NORM = "bn"
    ACTIVATION = nn.PReLU
    BATCH_SIZE = 8


class nnUNetTrainer_500ep_GN_PReLU_bs8(_BatchSizeBaseTrainer):
    NORM = "gn"
    ACTIVATION = nn.PReLU
    BATCH_SIZE = 8


class nnUNetTrainer_500ep_BN_Swish_bs8(_BatchSizeBaseTrainer):
    NORM = "bn"
    ACTIVATION = nn.SiLU
    BATCH_SIZE = 8


class nnUNetTrainer_500ep_GN_Swish_bs8(_BatchSizeBaseTrainer):
    NORM = "gn"
    ACTIVATION = nn.SiLU
    BATCH_SIZE = 8


# ===========================================================================
# BATCH A -- batch 4, 250 iterations. Fallback if batch 8 will not fit.
#            ~24 h/fold (2x data per epoch, optimiser steps unchanged).
# ===========================================================================
class nnUNetTrainer_500ep_BN_LeakyReLU_bs4(_BatchSizeBaseTrainer):
    NORM = "bn"
    BATCH_SIZE = 4
    HOLD_SAMPLES_PER_EPOCH = False


class nnUNetTrainer_500ep_GN_LeakyReLU_bs4(_BatchSizeBaseTrainer):
    NORM = "gn"
    BATCH_SIZE = 4
    HOLD_SAMPLES_PER_EPOCH = False


# ===========================================================================
# BATCH B -- batch 8, 250 iterations kept. Opposite-confound bracket:
#            4x data per epoch, optimiser steps unchanged. ~49 h/fold.
#            Run only if BN and GN both improve under Batch C.
# ===========================================================================
class nnUNetTrainer_500ep_BN_LeakyReLU_bs8_fulliter(_BatchSizeBaseTrainer):
    NORM = "bn"
    BATCH_SIZE = 8
    HOLD_SAMPLES_PER_EPOCH = False


class nnUNetTrainer_500ep_GN_LeakyReLU_bs8_fulliter(_BatchSizeBaseTrainer):
    NORM = "gn"
    BATCH_SIZE = 8
    HOLD_SAMPLES_PER_EPOCH = False


# ---------------------------------------------------------------------------
# Self-check:  python3 nnUNetTrainer_batchsize.py
# Prints every trainer with its settings and relative cost. Does not need a GPU.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H = 12.15   # measured h/fold at batch 2, from the ReLU fold-0 log
    print(f"{'trainer':46s} {'norm':4s} {'activation':11s} {'bs':>3s} "
          f"{'iters':>6s} {'samp/ep':>8s} {'rel':>6s} {'h/fold':>7s}")
    print("-" * 100)
    for name, obj in sorted(globals().items()):
        if not (name.startswith("nnUNetTrainer_500ep_") and isinstance(obj, type)):
            continue
        bs = obj.BATCH_SIZE
        it = (max(1, round(BASE_TRAIN_ITERS * BASE_BATCH / bs))
              if obj.HOLD_SAMPLES_PER_EPOCH else BASE_TRAIN_ITERS)
        rel = (bs * it) / (BASE_BATCH * BASE_TRAIN_ITERS)
        act = getattr(obj.ACTIVATION, "__name__", "LeakyReLU*")
        print(f"{name:46s} {obj.NORM.upper():4s} {act:11s} {bs:3d} "
              f"{it:6d} {bs*it:8d} {rel:5.2f}x {rel*H:6.1f}h")
    print("\n* LeakyReLU = nnU-Net default, no activation swap applied")
    print(f"reference: batch {BASE_BATCH} x {BASE_TRAIN_ITERS} iters "
          f"= {BASE_BATCH*BASE_TRAIN_ITERS} samples/epoch @ {H} h/fold")
