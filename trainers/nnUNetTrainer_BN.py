"""
Batch Normalisation configuration for the nnU-Net PlainConvUNet.

Replaces nnU-Net's InstanceNorm3d with BatchNorm3d (affine=True) and, in the
activation variants, replaces the default LeakyReLU. Both substitutions are
performed inside `build_network_architecture`, which runs BEFORE
`super().initialize()` constructs the optimiser, so the BatchNorm affine
parameters (gamma, beta) and the PReLU slope (alpha) are present in
`self.network.parameters()` when the optimiser is built and are optimised
normally.

Training length: `self.num_epochs = 500` is assigned after
`super().initialize()`, so the PolyLRScheduler is built on nnU-Net's default
1000-epoch horizon and training stops at epoch 500. The same ordering is used
in every trainer in this repository so that the configurations stay comparable.

API note (nnU-Net v2.2.1): `build_network_architecture` is called on the
trainer CLASS, not on an instance, so it is a @classmethod here and the
activation swap dispatches through `cls`. Do NOT convert it to a
@staticmethod (that loses the subclass identity, so every variant would build
with LeakyReLU) or to a plain instance method (that breaks the class-level
call).
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

    The clamp at x=20 prevents FP16 overflow inside torch.exp under nnU-Net's default
    mixed-precision training, without changing the function value (tanh(exp(x))
    saturates to 1 for x > ~4). The original BN trainer used the unclamped form, which
    is a latent AMP NaN risk; the clamp matches custom_brats_GN.py.
    """
    def forward(self, x):
        return x * torch.tanh(torch.exp(torch.clamp(x, max=20.0)))


# ===========================================================================
# Base trainer
# ===========================================================================

class BaseAdvancedTrainer(nnUNetTrainer):
    """
    PlainConvUNet + BatchNorm + 500-epoch training.

    The IN->BN substitution is common to all subclasses; subclasses override
    `_swap_activations` to replace nnU-Net's default LeakyReLU. BOTH swaps are
    performed inside `build_network_architecture`, BEFORE the optimiser is constructed
    by super().initialize(), so all swapped-in parameters (BatchNorm gamma/beta; PReLU
    alpha) enter the optimiser's parameter groups and are optimised normally.

    NOTE on LR schedule: self.num_epochs is set AFTER super().initialize(), so the
    PolyLRScheduler is built on nnU-Net's default 1000-epoch horizon even though
    training stops at epoch 500. The same ordering is used in the Instance Norm and
    Group Norm trainers, which is what keeps the cells comparable.
    """

    def initialize(self):
        super().initialize()
        self.num_epochs = 500  # see class docstring — order is deliberate

    # NOTE: @classmethod (not @staticmethod) — see the API note in the module
    # docstring. `cls` carries the variant identity so the correct activation swap
    # is dispatched via the MRO.
    @classmethod
    def build_network_architecture(cls, plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision=True):
        # Build the default PlainConvUNet (InstanceNorm + LeakyReLU) — the same network
        # as Basic-IN before any modification. The parent builder is a staticmethod, so
        # call it on nnUNetTrainer directly with positional args.
        network = nnUNetTrainer.build_network_architecture(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, enable_deep_supervision)

        # Swaps happen BEFORE returning -> before the optimiser snapshot in
        # super().initialize(). If these swaps are moved, re-check that
        # configure_optimizers still sees the final parameter set.
        network = cls._swap_activations(network)
        network = cls._swap_norms_to_bn(network)
        return network

    # ----- normalisation swap (common to all four subclasses) --------------

    @staticmethod
    def _swap_norms_to_bn(network):
        # Collect-then-mutate: never modify a module tree mid-iteration.
        targets = []
        for name, m in network.named_modules():
            if isinstance(m, (nn.InstanceNorm3d, nn.InstanceNorm2d, nn.InstanceNorm1d)):
                targets.append((name, m.num_features))
        for name, num_features in targets:
            parent_name, _, child_name = name.rpartition('.')
            parent = network.get_submodule(parent_name) if parent_name else network
            # affine=True -> gamma/beta are trainable. Because this runs before the
            # optimiser is built, they enter the optimiser's parameter groups.
            setattr(parent, child_name, nn.BatchNorm3d(num_features, affine=True))
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
# Activation variants
# Class names are retained for drop-in compatibility with existing checkpoints.
# "BN_" aliases are provided at the bottom.
# ===========================================================================

class nnUNetTrainer_Adv_LeakyReLU(BaseAdvancedTrainer):
    """BN baseline: nnU-Net's default LeakyReLU is retained; only the norm differs."""
    pass


class nnUNetTrainer_Adv_ReLU(BaseAdvancedTrainer):
    @classmethod
    def _swap_activations(cls, network):
        return cls._replace_leakyrelu_with(network, lambda: nn.ReLU(inplace=True))


class nnUNetTrainer_Adv_Swish(BaseAdvancedTrainer):
    @classmethod
    def _swap_activations(cls, network):
        return cls._replace_leakyrelu_with(network, lambda: nn.SiLU(inplace=True))


class nnUNetTrainer_Adv_PReLU(BaseAdvancedTrainer):
    """
    Single shared learnable slope: nn.PReLU() -> num_parameters=1, init=0.25, matching
    the PReLU configuration used in the Instance Norm and Group Norm trainers.
    """
    @classmethod
    def _swap_activations(cls, network):
        return cls._replace_leakyrelu_with(network, lambda: nn.PReLU())


class nnUNetTrainer_Adv_TanhExp(BaseAdvancedTrainer):
    @classmethod
    def _swap_activations(cls, network):
        return cls._replace_leakyrelu_with(network, lambda: TanhExp())


# ===========================================================================
# Aliases
# Exposing both spellings keeps existing checkpoints loadable (resolved by the name
# stored at training time) while letting new runs use the BN_ names. nnU-Net resolves
# trainers by name, so both point at the same class.
# ===========================================================================
nnUNetTrainer_500ep_BN_LeakyReLU = nnUNetTrainer_Adv_LeakyReLU
nnUNetTrainer_500ep_BN_ReLU      = nnUNetTrainer_Adv_ReLU
nnUNetTrainer_500ep_BN_Swish     = nnUNetTrainer_Adv_Swish
nnUNetTrainer_500ep_BN_PReLU     = nnUNetTrainer_Adv_PReLU
nnUNetTrainer_500ep_BN_TanhExp   = nnUNetTrainer_Adv_TanhExp
