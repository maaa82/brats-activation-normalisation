import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class TanhExp(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))

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

class nnUNetTrainer_500ep_LeakyReLU(BaseAblationTrainer): pass

class nnUNetTrainer_500ep_Swish(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, nn.SiLU)

class nnUNetTrainer_500ep_TanhExp(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, TanhExp)

class nnUNetTrainer_500ep_PReLU(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, nn.PReLU)

class nnUNetTrainer_500ep_ReLU(BaseAblationTrainer):
    def build_network_architecture(self, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision)
        return self.swap_activations(network, nn.ReLU)