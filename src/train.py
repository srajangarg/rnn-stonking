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

from .eval import evaluate

from .data import get_dataloader, get_valdataloader
from .utils.tb_visualizer import TBVisualizer

torch.backends.cudnn.benchmark = True

torch.multiprocessing.set_sharing_strategy('file_system')

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="dummy")
def main(cfg : DictConfig) -> None:
    log.info('###### Config ######')
    log.info(OmegaConf.to_yaml(cfg))

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    log.info('###### Loading data ######')
    dataloader = get_dataloader(cfg)
    log.info(dataloader)
    log.info(dataloader.dataset)
    log.info('###########################')

    # log.info('###### Loading val data ######')
    # val_dataloader = get_valdataloader(cfg)
    # log.info(val_dataloader)
    # log.info(val_dataloader.dataset)
    # log.info('###########################')

    log.info('###### Creating model #####')
    model: torch.nn.Module = instantiate(cfg.model)
    log.info(model)
    log.info('###########################')

    log.info('###### Creating crit ######')
    criterion = instantiate(cfg.criterion)
    log.info(criterion)
    log.info('###########################')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    log.info(f'using device {device}')

    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    viz = TBVisualizer(cfg.display.logdir)

    iter = 0
    for epoch in range(cfg.optim.num_epochs):

        # Evaluate
        if epoch % 10 == 0:
            evaluate(model, dataloader, epoch, output_dir='eval/', viz=viz)

        for i, inputs in enumerate(dataloader):
            inputs = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in inputs.items()}

            # Zero the optimizer gradient.
            optimizer.zero_grad()

            # Calculate positions and loss
            positions = model(inputs)
            loss, metrics = criterion(positions, inputs)

            # Take the training step.
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            if (iter%cfg.display.iprint)==0:
                with torch.no_grad():
                    float_metrics = {k:v for k,v in metrics.items() if isinstance(v, float)}
                    float_metrics['grad_norm'] = grad_norm

                    viz.plot_current_scalars(float_metrics, iter)

                    s = f'e{epoch:4d};  iter{iter:6d};  loss{loss:6.2f};  '
                    s = s + ';  '.join([f'{k}{v:9.1e}' for k,v in float_metrics.items()])
                    log.info(s)

            iter += 1

        # if (epoch%cfg.display.ival)==0:
        #     # Evaluate
        #     evaluate(model, val_dataloader, epoch, output_dir='eval/', viz=viz)

if __name__ == "__main__":
    main()
