"""
nnUNetTrainer_bench12.py  --  REAL-EPOCH training-cost benchmark.

Twelve trainers, one per activation function, all under nnU-Net's BASIC
configuration (Instance Normalisation, batch 2, 250 iters/epoch, deep
supervision, default dataloader). Each trains fold 0 for BENCH_EPOCHS real
epochs on the real BraTS 2023 preprocessed data, so epoch time includes
dataloading and augmentation exactly as in the production runs.

What is measured per epoch and written to the training log:
    [bench] epoch N | epoch_time_s | peak_alloc_GB | peak_reserved_GB
Peak memory is torch.cuda.max_memory_allocated / max_memory_reserved over the
epoch (reset at epoch start), i.e. the REAL process footprint including
optimiser state and activations -- not the model-only figure from
vram_profile_grid.py.

Deliberately NOT done: the end-of-training validation prediction over the
fold's ~250 cases (perform_actual_validation is a no-op) -- it would add
30-60 min per activation and measures inference, not training.

Self-contained: does not import from any other custom trainer file, so a
broken sibling module cannot take it down. Class names are unique
(nnUNetTrainer_bench_IN_<Act>) and cannot collide with the production
trainers. Copy ONLY this file into nnU-Net's trainer directory.

Cost: BENCH_EPOCHS=10 x ~85 s = ~15 min per activation; 12 activations on
5 GPUs = 3 waves = ~45 min total. Use epochs 1..9 for the mean (epoch 0 is
warm-up: cudnn autotune + dataloader spin-up).
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

BENCH_EPOCHS = 10           # override with env BENCH_EPOCHS if you like
import os
BENCH_EPOCHS = int(os.environ.get("BENCH_EPOCHS", BENCH_EPOCHS))


# ---------------------------------------------------------------------------
# Activations -- byte-identical to the definitions in the trainer files
# ---------------------------------------------------------------------------
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

class _ReLU(nn.ReLU):
    def __init__(self): super().__init__(inplace=True)

class _ELU(nn.ELU):
    def __init__(self): super().__init__(inplace=True)

class _GELU(nn.GELU):
    def __init__(self): super().__init__()

class _Swish(nn.SiLU):
    def __init__(self): super().__init__(inplace=True)

class _PReLU(nn.PReLU):
    # single shared learnable slope, init 0.25 -- matches the production PReLU cell
    def __init__(self): super().__init__(num_parameters=1, init=0.25)


# ---------------------------------------------------------------------------
class _Bench12Base(nnUNetTrainer):
    ACTIVATION = None      # None = keep nnU-Net's LeakyReLU

    @staticmethod
    def _swap_act(network, new_cls):
        n = 0
        def rec(m):
            nonlocal n
            for name, child in m.named_children():
                if isinstance(child, nn.LeakyReLU):
                    setattr(m, name, new_cls()); n += 1
                else:
                    rec(child)
        rec(network)
        return network, n

    @classmethod
    def build_network_architecture(cls, plans_manager, dataset_json,
                                   configuration_manager, num_input_channels,
                                   enable_deep_supervision=True):
        net = nnUNetTrainer.build_network_architecture(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, enable_deep_supervision)
        if cls.ACTIVATION is not None:
            net, n = cls._swap_act(net, cls.ACTIVATION)
            assert n == 44, f"expected 44 activation modules, replaced {n}"
        return net

    def initialize(self):
        super().initialize()
        # set AFTER super().initialize(), same as every production trainer, so
        # the LR schedule is built the same way (irrelevant for timing, but
        # keeps the run byte-comparable in every other respect)
        self.num_epochs = BENCH_EPOCHS
        act = getattr(self.ACTIVATION, "__name__", "LeakyReLU(default)").lstrip("_")
        self.print_to_log_file(
            f"[bench] activation={act} norm=IN batch={self.configuration_manager.batch_size} "
            f"iters/epoch={self.num_iterations_per_epoch} val_iters={self.num_val_iterations_per_epoch} "
            f"epochs={self.num_epochs} gpu={torch.cuda.get_device_name(0)}")

    # ---- per-epoch instrumentation ---------------------------------------
    def on_epoch_start(self):
        super().on_epoch_start()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        self._bench_t0 = time.time()

    def on_epoch_end(self):
        super().on_epoch_end()
        torch.cuda.synchronize()
        dt = time.time() - self._bench_t0
        alloc = torch.cuda.max_memory_allocated() / 1e9
        resv  = torch.cuda.max_memory_reserved() / 1e9
        # current_epoch was already incremented by super().on_epoch_end()
        self.print_to_log_file(
            f"[bench] epoch {self.current_epoch - 1} | epoch_time_s {dt:.2f} | "
            f"peak_alloc_GB {alloc:.2f} | peak_reserved_GB {resv:.2f}")

    # ---- skip the expensive end-of-training validation prediction --------
    def perform_actual_validation(self, save_probabilities: bool = False):
        self.print_to_log_file("[bench] perform_actual_validation skipped (benchmark run)")


# ---------------------------------------------------------------------------
# The twelve trainers (basic nnU-Net = Instance Norm)
# ---------------------------------------------------------------------------
class nnUNetTrainer_bench_IN_LeakyReLU(_Bench12Base): ACTIVATION = None
class nnUNetTrainer_bench_IN_ReLU(_Bench12Base):      ACTIVATION = _ReLU
class nnUNetTrainer_bench_IN_PReLU(_Bench12Base):     ACTIVATION = _PReLU
class nnUNetTrainer_bench_IN_ELU(_Bench12Base):       ACTIVATION = _ELU
class nnUNetTrainer_bench_IN_GELU(_Bench12Base):      ACTIVATION = _GELU
class nnUNetTrainer_bench_IN_Swish(_Bench12Base):     ACTIVATION = _Swish
class nnUNetTrainer_bench_IN_Mish(_Bench12Base):      ACTIVATION = Mish
class nnUNetTrainer_bench_IN_ELiSH(_Bench12Base):     ACTIVATION = ELiSH
class nnUNetTrainer_bench_IN_HardELiSH(_Bench12Base): ACTIVATION = HardELiSH
class nnUNetTrainer_bench_IN_TanhExp(_Bench12Base):   ACTIVATION = TanhExp
class nnUNetTrainer_bench_IN_Logish(_Bench12Base):    ACTIVATION = Logish
class nnUNetTrainer_bench_IN_Smish(_Bench12Base):     ACTIVATION = Smish

BENCH12 = ["LeakyReLU", "ReLU", "PReLU", "ELU", "GELU", "Swish",
           "Mish", "ELiSH", "HardELiSH", "TanhExp", "Logish", "Smish"]

if __name__ == "__main__":
    import sys
    print(f"BENCH_EPOCHS={BENCH_EPOCHS}")
    for a in BENCH12:
        c = globals()[f"nnUNetTrainer_bench_IN_{a}"]
        print(f"  {c.__name__:34s} activation={getattr(c.ACTIVATION,'__name__','LeakyReLU(default)').lstrip('_')}")
