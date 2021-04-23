import os

import matplotlib as mpl

if 'DISPLAY' not in os.environ:
    print('Display not found. Using Agg backend for matplotlib')
    mpl.use('Agg')
else:
    print(f'Found display : {os.environ["DISPLAY"]}')
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import tensorboardX
import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataloader import DataLoader

from .criterions import SharpeCriterion
from .utils.tb_visualizer import TBVisualizer

# A logger for this file
log = logging.getLogger(__name__)

@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, epoch: int,
        output_dir: Optional[str]=None, viz: Optional[TBVisualizer]=None,
        device = 'cuda:0'):
    pass

    log.info(f'Evaluating model at epoch {epoch}')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    criterion = SharpeCriterion().to(device)

    MOVE = []
    VOLA = []
    PNL = []
    DATE = []
    metrics_list = []
    positions_dict = {}

    for i, inputs in enumerate(dataloader):
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in inputs.items()}

        # Calculate positions and loss
        positions = model(inputs)
        loss, metrics = criterion(positions, inputs)

        # Note: Assuming batch_size = 1
        assert(inputs['price_raw'].shape[0] == 1)

        # Calculate metrics
        day_open = inputs['price_raw'][0, 0]  # Close of first minute that day
        day_close = inputs['price_raw'][0, -1]  # Close of last minute that day
        movement = day_close - day_open
        volatility = inputs['price_raw'].std()/inputs['price_raw'].mean()
        raw_pnl = metrics['pnl_sum'] * inputs['price_raw'][0,0].item()

        metrics_list.append(
            dict(
                **{k:v for k,v in metrics.items() if isinstance(v, float)},
                volatility=volatility,
                movement=movement,
                raw_pnl=raw_pnl,
                datetime=inputs["datetime"][0],
            )
        )

        positions_dict.update({inputs["datetime"][0]: positions.cpu()})
    positions_dump_file = output_dir/f'positions_dict_e{epoch:03d}.pth'
    positions_dump_file.parent.mkdir(exist_ok=True)
    torch.save(positions_dict, positions_dump_file)

    df = pd.DataFrame(metrics_list).set_index('datetime').sort_index()

    # Scatter plots
    fig = plt.figure()
    plt.scatter(df['movement'], df['pnl_sum'])
    pnl_vs_move = tensorboardX.utils.figure_to_image(fig)
    plt.close()
    viz.plot_images({'pnl_vs_move': pnl_vs_move}, epoch)

    # Scatter plots
    fig = plt.figure()
    plt.scatter(df['volatility'], df['pnl_sum'])
    pnl_vs_vola = tensorboardX.utils.figure_to_image(fig)
    plt.close()
    viz.plot_images({'pnl_vs_vola': pnl_vs_vola}, epoch)

    # Line plots
    fig = plt.figure()
    plt.plot(df.index, df['pnl_sum'].cumsum())
    pnl_vs_time = tensorboardX.utils.figure_to_image(fig)
    plt.close()
    viz.plot_images({'pnl_vs_time': pnl_vs_time}, epoch)

    # Line plots
    fig = plt.figure()
    plt.hist(df['pnl_sum'])
    pnl_hist = tensorboardX.utils.figure_to_image(fig)
    plt.close()
    viz.plot_images({'pnl_hist': pnl_hist}, epoch)

    log.info(f'Done evaluating model at epoch {epoch}')
