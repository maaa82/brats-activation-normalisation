"""
Group Normalisation configuration for the nnU-Net PlainConvUNet.

Architecture : PlainConvUNet (nnU-Net default, unchanged)
Normalisation: GroupNorm, 8 groups (replaces nnU-Net's InstanceNorm)
Activations  : LeakyReLU (baseline), PReLU, Swish (SiLU), TanhExp
Epochs       : 500

The IN -> GN substitution and the activation swap are both performed inside
`build_network_architecture`, BEFORE the optimiser is constructed by
`super().initialize()`, so GroupNorm's learnable affine parameters and any
activation parameters are registered with the optimiser.

No re-planning is required, because the architecture is unchanged: the existing
plans file applies. Use the same 5-fold splits as the other normalisation
configurations so that cells can be compared pairwise.

Training:
    nnUNetv2_train <DATASET_ID> 3d_fullres <FOLD> -tr nnUNetTrainer_500ep_GN_LeakyReLU
    nnUNetv2_train <DATASET_ID> 3d_fullres <FOLD> -tr nnUNetTrainer_500ep_GN_PReLU
    nnUNetv2_train <DATASET_ID> 3d_fullres <FOLD> -tr nnUNetTrainer_500ep_GN_Swish
    nnUNetv2_train <DATASET_ID> 3d_fullres <FOLD> -tr nnUNetTrainer_500ep_GN_TanhExp

API note (nnU-Net v2.2.1): `build_network_architecture` is called on the
trainer CLASS, not on an instance:

    network = trainer_class.build_network_architecture(plans_manager, ...)

Because the per-variant activation choice must be polymorphic, the method is a
@classmethod so that `cls` is bound to the variant subclass at call time and
`cls._swap_*` resolves to the correct override via the MRO. The swap helpers
are classmethods/staticmethods for the same reason. Do NOT change these to
plain instance methods (`self`), which break the class-level call, or to a
@staticmethod, which silently loses the subclass identity and would build every
variant with the base LeakyReLU.
"""

import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# ===========================================================================
# Custom modules
# ===========================================================================

class TanhExp(nn.Module):
    """
    Numerically stable TanhExp = x * tanh(exp(x)).

    The clamp at x=20 avoids FP16/FP32 overflow inside torch.exp without
    changing the function value: tanh(exp(20)) = 1 to within ~10^-200, and
    tanh(exp(x)) saturates to 1 for x > ~4 anyway. Without this clamp,
    nnU-Net's default mixed-precision training NaNs out within the first
    few hundred iterations because FP16's max is ~65504 and exp(11.1) already
    exceeds it.
    """
    def forward(self, x):
        return x * torch.tanh(torch.exp(torch.clamp(x, max=20.0)))


class GroupNorm3d(nn.GroupNorm):
    """
    GroupNorm with an InstanceNorm3d / BatchNorm3d-style constructor signature:
    takes num_features as the first positional argument so it can be swapped
    in transparently by the same factory code that handles IN/BN.

    Picks the largest valid num_groups <= the target so it never crashes on
    odd channel counts. For nnU-Net's default channel sequence
    (32/64/128/256/320) num_groups=8 divides cleanly at every stage.
    """
    def __init__(self, num_features, num_groups=8, eps=1e-5, affine=True):
        g = min(num_groups, num_features)
        while num_features % g != 0:
            g -= 1
        super().__init__(num_groups=g, num_channels=num_features,
                         eps=eps, affine=affine)


# ===========================================================================
# Base trainer
# ===========================================================================

class BaseGNTrainer(nnUNetTrainer):
    """
    PlainConvUNet + GroupNorm + 500-epoch training.

    Subclasses override `_swap_activations` to replace nnU-Net's default
    LeakyReLU. The IN -> GN substitution is common to all subclasses and
    happens inside `build_network_architecture` (BEFORE the optimiser is
    constructed by super().initialize()), so GN's learnable affine
    parameters end up in the optimiser's parameter groups.

    NOTE on LR schedule: self.num_epochs is set AFTER super().initialize(),
    so the PolyLRScheduler is built on nnU-Net's default 1000-epoch horizon
    even though training stops at epoch 500. Matching this ordering across the
    Instance Norm, Batch Norm and Group Norm trainers is what makes the
    controlled comparison valid.
    """

    NUM_GROUPS = 8

    def initialize(self):
        super().initialize()
        self.num_epochs = 500  # see class docstring — order is deliberate

    # NOTE: @classmethod (not @staticmethod) — see API-COMPATIBILITY NOTE at
    # the top of the file. `cls` carries the variant identity so the correct
    # activation swap is dispatched.
    @classmethod
    def build_network_architecture(cls, plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision=True):
        # Build the default PlainConvUNet (the same network as the Instance Norm
        # configuration before any modification). The parent builder is a staticmethod, so
        # call it on nnUNetTrainer directly with positional args.
        network = nnUNetTrainer.build_network_architecture(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, enable_deep_supervision)

        # IMPORTANT: do swaps BEFORE returning. The optimiser is constructed
        # later inside super().initialize() via self.network.parameters(),
        # which will pick up the GroupNorm and any new activation parameters
        # only if they are present at that point. If these swaps are moved
        # elsewhere, re-check that configure_optimizers still sees the final
        # parameter set.
        #
        # cls._swap_activations dispatches to the variant override (PReLU /
        # Swish / TanhExp), or the base no-op for the LeakyReLU baseline.
        network = cls._swap_activations(network)
        network = cls._swap_norms_to_gn(network)
        return network

    # ----- normalisation swap (common to all four subclasses) --------------

    @classmethod
    def _swap_norms_to_gn(cls, network):
        # Collect-then-mutate: never modify a module tree mid-iteration.
        targets = []
        for name, m in network.named_modules():
            if isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d,
                              nn.InstanceNorm2d, nn.BatchNorm2d,
                              nn.InstanceNorm1d, nn.BatchNorm1d)):
                targets.append((name, m.num_features))
        for name, num_features in targets:
            parent_name, _, child_name = name.rpartition('.')
            parent = network.get_submodule(parent_name) if parent_name else network
            setattr(parent, child_name,
                    GroupNorm3d(num_features, num_groups=cls.NUM_GROUPS))
        return network

    # ----- activation swap (overridden per subclass) -----------------------

    @classmethod
    def _swap_activations(cls, network):
        """Default: keep nnU-Net's LeakyReLU. Subclasses override."""
        return network

    @staticmethod
    def _replace_leakyrelu_with(network, factory):
        targets = [name for name, m in network.named_modules()
                   if isinstance(m, nn.LeakyReLU)]
        for name in targets:
            parent_name, _, child_name = name.rpartition('.')
            parent = network.get_submodule(parent_name) if parent_name else network
            setattr(parent, child_name, factory())
        return network


# ===========================================================================
# Four activation variants
# ===========================================================================

class nnUNetTrainer_500ep_GN_LeakyReLU(BaseGNTrainer):
    """
    GN baseline. nnU-Net's default LeakyReLU(negative_slope=0.01, inplace=True)
    is retained; only the normalisation differs from Basic-LeakyReLU.

    This is the reference cell for the three-way normalisation comparison
    at a fixed activation: IN-LeakyReLU <-> BN-LeakyReLU <-> GN-LeakyReLU.
    """
    pass


class nnUNetTrainer_500ep_GN_PReLU(BaseGNTrainer):
    """
    GN + PReLU. Single learnable alpha (num_parameters=1, init=0.25), matching
    the PReLU configuration in the Instance Norm and Batch Norm trainers
    (all three use the bare nn.PReLU constructor).

    Note: this is the single-alpha variant, not per-channel PReLU. It matches
    the PReLU configuration used in the Instance Norm and Batch Norm trainers,
    which is what makes the cells directly comparable. For true per-channel
    PReLU (one alpha per feature map), the factory becomes
        lambda: nn.PReLU(num_parameters=<channels>, init=0.25)
    and a sibling-aware variant of _replace_leakyrelu_with is required, one
    that reads the channel count from the preceding Conv3d or InstanceNorm3d.
    """
    @classmethod
    def _swap_activations(cls, network):
        return cls._replace_leakyrelu_with(network, lambda: nn.PReLU())


class nnUNetTrainer_500ep_GN_Swish(BaseGNTrainer):
    """
    GN + Swish. Uses nn.SiLU() (inplace=False), matching the Instance Norm
    and Batch Norm Swish cells.

    This cell completes the Swish row of the normalisation comparison, which
    distinguishes an interaction specific to batch-level statistics from one
    that holds for any non-instance normalisation.
    """
    @classmethod
    def _swap_activations(cls, network):
        return cls._replace_leakyrelu_with(network, lambda: nn.SiLU())


class nnUNetTrainer_500ep_GN_TanhExp(BaseGNTrainer):
    """
    GN + TanhExp. Uses the clamped-input variant defined above for AMP/FP16
    stability: the unclamped x * tanh(exp(x)) overflows FP16 above x ~= 11,
    which causes early-training NaNs under nnU-Net's default mixed-precision
    setting.
    """
    @classmethod
    def _swap_activations(cls, network):
        return cls._replace_leakyrelu_with(network, lambda: TanhExp())


# ===========================================================================
# Spelling alias
# ===========================================================================
# The results folder for the PReLU variant is spelled "...GN_PreLU...", while
# the canonical class above is "...GN_PReLU". nnU-Net resolves the trainer by
# the name stored in the checkpoint, so both spellings are exposed as the same
# class, which makes inference robust regardless of which was used at training
# time.
nnUNetTrainer_500ep_GN_PreLU = nnUNetTrainer_500ep_GN_PReLU
