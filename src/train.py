"""
python -m src.train
"""

from __future__ import absolute_import, division, print_function

import logging
import random

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .data import get_dataloader
from .utils.tb_visualizer import TBVisualizer

torch.backends.cudnn.benchmark = True

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="dummy")
def main(cfg : DictConfig) -> None:
    print('###### Config ######')
    print(OmegaConf.to_yaml(cfg))

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    print('###### Loading data ######')
    dataloader = get_dataloader(cfg)
    print(dataloader)
    print(dataloader.dataset)
    print('###########################')

    print('###### Creating model #####')
    model = instantiate(cfg.model)
    print(model)
    print('###########################')

    print('###### Creating crit ######')
    criterion = instantiate(cfg.criterion)
    print(criterion)
    print('###########################')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('using device', device)

    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    viz = TBVisualizer(cfg.display.logdir)

    iter = 0
    for epoch in range(cfg.optim.num_epochs):
        for i, inputs in enumerate(dataloader):
            inputs = {k:v.to(device) for k,v in inputs.items()}

            # Zero the optimizer gradient.
            optimizer.zero_grad()

            # Calculate positions and loss
            positions = model(inputs)
            loss, metrics = criterion(positions, inputs)

            # Take the training step.
            loss.backward()
            optimizer.step()

            if (iter%cfg.display.iprint)==0:
                with torch.no_grad():
                    float_metrics = {k:v for k,v in metrics.items() if isinstance(v, float)}
                    viz.plot_current_scalars(float_metrics, iter)

                    s = f'e{epoch:4d};  iter{iter:6d};  loss{loss:6.2f};  '
                    s = s + ';  '.join([f'{k}{v:9.1e}' for k,v in float_metrics.items()])
                    log.info(s)

            iter += 1


if __name__ == "__main__":
    main()
