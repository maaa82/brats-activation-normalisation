import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# ---------------------------------------------------------------------------
# In-domain activation screen on the basic (Instance Norm) nnU-Net.
# Companion to nnUNetTrainer_IN.py, which already defines LeakyReLU, ReLU,
# PReLU, Swish and TanhExp. This file adds the remaining activations of the
# twelve. ReLU is NOT redefined here — duplicating the class name across two
# files in the trainer search path makes nnU-Net's class lookup ambiguous.
#
# Activation definitions are kept identical to those used in the companion
# experiments so that the same functions are being compared. Do not rewrite
# them (e.g. log1p for log(1+.)) — comparability depends on it.
# ---------------------------------------------------------------------------

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


class BaseAblationTrainer(nnUNetTrainer):
    def initialize(self):
        # 1. Let the parent build everything normally first
        super().initialize()

        # 2. Set the epoch limit used throughout this study
        self.num_epochs = 500

        # 3. Set the worker limit to 4 per GPU
        self.num_processes_training = 4

    def swap_activations(self, network, new_activation_class):
        def replace(module):
            for name, child in module.named_children():
                if isinstance(child, nn.LeakyReLU):
                    setattr(module, name, new_activation_class())
                else:
                    replace(child)
        replace(network)
        return network


class nnUNetTrainer_500ep_ELU(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, nn.ELU)

class nnUNetTrainer_500ep_GELU(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, nn.GELU)

class nnUNetTrainer_500ep_Mish(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, Mish)

class nnUNetTrainer_500ep_ELiSH(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, ELiSH)

class nnUNetTrainer_500ep_HardELiSH(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, HardELiSH)

class nnUNetTrainer_500ep_Logish(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, Logish)

class nnUNetTrainer_500ep_Smish(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, Smish)
