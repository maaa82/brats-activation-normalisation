"""
Control cell: Batch Norm + PReLU with the activation slope held fixed.

    BatchNorm affine (gamma, beta): TRAINABLE
    PReLU slope (alpha)           : FROZEN at 0.25

Identical to nnUNetTrainer_Adv_PReLU in every other respect. Holding alpha
fixed while the affine trains separates the contribution of the learnable
slope from the contribution of the normalisation affine, which a comparison
between a fully trained cell and a fully fixed one cannot do.

Deployment
----------
Place this file next to the Batch Norm trainer in
    .../nnunetv2/training/nnUNetTrainer/
It imports BaseAdvancedTrainer from nnUNetTrainer_Advanced, so that module must
be installed under that name. It defines exactly one new class and redefines
nothing, so there is no class-name collision with the other trainers.

Train (5 folds, as for the other Batch Norm cells):
    python3 -m nnunetv2.run.run_training 100 3d_fullres <FOLD> \
        -tr nnUNetTrainer_Adv_PReLU_FrozenAlpha --npz

Results are written to a distinct folder,
    nnUNetTrainer_Adv_PReLU_FrozenAlpha__nnUNetPlans__3d_fullres
so the trained BN-PReLU cell is never overwritten.

Why requires_grad=False is the correct freeze
---------------------------------------------
The swap runs inside build_network_architecture, BEFORE the optimiser is built,
so alpha IS handed to SGD. With requires_grad=False its gradient stays None and
SGD skips it (`if p.grad is None: continue`), so alpha never moves from 0.25.
The BatchNorm affine parameters keep requires_grad=True and train normally.
This is a declared freeze of a single named parameter.
"""

from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_Advanced import BaseAdvancedTrainer


def _make_frozen_prelu():
    """Single shared PReLU with alpha fixed at 0.25 (not optimised)."""
    act = nn.PReLU(num_parameters=1, init=0.25)
    act.weight.requires_grad_(False)   # freeze alpha; BN affine is untouched and stays trainable
    return act


class nnUNetTrainer_Adv_PReLU_FrozenAlpha(BaseAdvancedTrainer):
    """
    BN + PReLU with TRAINABLE affine and FROZEN alpha (0.25).

    Identical to nnUNetTrainer_Adv_PReLU in every respect except that the PReLU slope is
    held at its 0.25 initialisation. This is a control cell, not one of the four
    Batch Norm activation cells.
    """
    @classmethod
    def _swap_activations(cls, network):
        return cls._replace_leakyrelu_with(network, _make_frozen_prelu)


# Alias mirroring the BN_ aliases in the Batch Norm trainer.
nnUNetTrainer_500ep_BN_PReLU_FrozenAlpha = nnUNetTrainer_Adv_PReLU_FrozenAlpha
