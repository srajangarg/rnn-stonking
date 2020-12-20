import inspect
import torch
from torch import nn

from .utils.misc import get_default_kwargs
from .utils import net_blocks

class SimpleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = create_rnn_backbone(cfg.backbone)    # Initiazte LSTM to process features
        self.readout = create_readout_network(cfg.readout, cfg.backbone.hidden_size)

    def forward(self, inputs):
        feats = inputs['feats']     # BxTxF
        b,t,f = feats.shape

        rnn_outputs, _ = self.backbone(feats)  # BxTxH
        positions = self.readout(rnn_outputs.view(b*t, -1)) # BTx1
        positions = positions.view(b,t,1)

        return positions

def create_rnn_backbone(cfg):
    """ Builds RNN backbone for processing sequence of features
    """
    if cfg.name.lower() == 'lstm':
        kwargs = get_default_kwargs(nn.RNN.__init__)
        kwargs.update({k:v for k,v in cfg.items() if k in kwargs})
        return nn.LSTM(cfg.input_size, cfg.hidden_size, **kwargs)
    else:
        raise NotImplementedError

def create_readout_network(cfg, input_size):
    modules = []
    if cfg.num_layers>0:
        modules.append(
            net_blocks.fc_stack(input_size,
                                cfg.hidden_size,
                                cfg.num_layers),
        )
        input_size = cfg.hidden_size
    modules.append(nn.Linear(input_size, cfg.output_size))
    modules.append(nn.Sigmoid())    # Note: positions are in (0,1)
    return nn.Sequential(*modules)

def get_model_class_by_name(name):
    """ Searches for nn.Module classes in this file with matching (case-insensitive) name
    """
    all_classes = {k.lower():v for k,v in globals().items() if (
                        inspect.isclass(v) and
                        (nn.Module in inspect.getmro(v))
                    )}
    return all_classes[name.lower()]

def create_model(cfg) -> nn.Module:
    model_class = get_model_class_by_name(cfg.name)
    return model_class(cfg)

