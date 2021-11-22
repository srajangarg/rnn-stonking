from typing import Tuple
import torch
from torch import nn

class SharpeCriterion(nn.Module):
    def __init__(self, loss_type='pnl'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, positions:torch.Tensor, inputs:dict) -> Tuple[torch.Tensor, dict]:
        # `positions` is a BxTx1 tensor containing the amount of stock held
        # at the end of a candlestick. Changes in position correspond to trades.

        # TODO: What price are these trades executed at? Currently assuming
        # the prices is set in the dataloader. Instead, we can pass a low/high
        # price range and use low for sells and high for buys to emulate worst
        # case scenario
        price = inputs['price']     # Bx(T+1)x1     Price at which stock is traded

        # TODO: set warmup - first 100 positions are zero

        # Add zero-position at start/end
        zero_pos = torch.zeros_like(positions[:,:1,:])

        #
        # positions[:,-1,:] = 0.0
        positions = torch.cat([zero_pos, positions, zero_pos], dim=1)
        delta_pos = positions[:,1:,:] - positions[:,:-1,:]      # Bx(T+1)x1 stock bought
        pnl_t = -1 * (delta_pos*price)                          # Note: negative sign

        # Normalizing price:
        # typical pnl when predicting shares is given by    \mean shares * price
        # we isntead predict shares (in money value) as     \mean (shares*mean_price) * (price/mean_price)

        pnl =  pnl_t.sum(dim=1)            # Bx1   pnl per candlestick
        # sharpe = pnl.mean() / (pnl.std() + 1e-8)

        metrics = {
            'pnl_t': pnl_t,
            'pnl_cumsum': torch.cumsum(pnl_t, dim=1)[:,0],     # accummulated pnl over time
            'pnl_sum': pnl_t.sum(dim=1).mean().item(),
            'pnl_mean': pnl.mean().item(),
            # 'pnl_std': pnl.std().item(),
            # 'sharpe': sharpe.item(),
        }

        # if self.loss_type == 'sharpe':
        #     loss = -1 * sharpe          # Maximize sharpe
        if self.loss_type == 'pnl':
            loss = -1 * pnl.mean()      # Maximize pnl
        else:
            raise ValueError(self.loss_type)

        return loss, metrics
