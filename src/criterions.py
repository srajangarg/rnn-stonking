import inspect
import torch
from torch import nn

class SharpeCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, positions, inputs):
        # `positions` is a BxTx1 tensor containing the amount of stock held
        # at the end of a candlestick. Changes in position correspond to trades.

        # TODO: What price are these trades executed at? Currently assuming
        # the prices is set in the dataloader. Instead, we can pass a low/high
        # price range and use low for sells and high for buys to emulate worst
        # case scenario
        price = inputs['price']     # BxTx1     Price at which stock is traded

        # Add zero-position at t+1
        positions = torch.cat([positions, torch.zeros_like(positions[:,:1,:])], dim=1)
        delta_pos = positions[:,1:,:] - positions[:,:-1,:]      # BxTx1 stock bought
        pnl_t = -1 * (delta_pos*price)                          # Note: negative sign

        pnl =  pnl_t.mean(dim=1)            # Bx1   pnl per candlestick
        sharpe = pnl.mean() / pnl.std()

        metrics = {
            'pnl': torch.cumsum(pnl_t, dim=1)[:,0],     # accummulated pnl over time
            'pnl_mean': pnl.mean(),
            'pnl_std': pnl.std(),
        }

        # Notice that loss is negative sharpe
        return -1 * sharpe, metrics


def get_model_class_by_name(name):
    """ Searches for nn.Module classes in this file with matching (case-insensitive) name
    """
    all_classes = {k.lower():v for k,v in globals().items() if (
                        inspect.isclass(v) and
                        (nn.Module in inspect.getmro(v))
                    )}
    return all_classes[name.lower()]

def create_criterion(cfg) -> nn.Module:
    model_class = get_model_class_by_name(cfg.name)
    return model_class(cfg)

