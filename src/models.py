import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from .utils import net_blocks

class SimpleModel(nn.Module):
    def __init__(self, backbone:nn.Module, readout:DictConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.readout = create_readout_network(readout, backbone.hidden_size)

    def forward(self, inputs:dict) -> torch.Tensor:
        feats = inputs['feats']     # BxTxF
        b,t,f = feats.shape

        rnn_outputs, _ = self.backbone(feats)   # BxTxH
        positions = self.readout(rnn_outputs.reshape(b*t, -1))  # BxTxA
        positions = positions.view(b,t,6)

        return positions

# TODO: SimpleNModel

def create_readout_network(cfg:DictConfig, input_size:int) -> nn.Module:
    modules = []
    if cfg.num_layers>0:
        modules.append(
            net_blocks.fc_stack(input_size,
                                cfg.hidden_size,
                                cfg.num_layers),
        )
        input_size = cfg.hidden_size
    modules.append(nn.Linear(input_size, cfg.output_size))
    modules.append(nn.Tanh())    # Note: positions are in (-1,1)
    return nn.Sequential(*modules)
