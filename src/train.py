"""
python -m src.train
"""

from __future__ import absolute_import, division, print_function

import os
import os.path as osp
import random

import matplotlib as mpl
if 'DISPLAY' not in os.environ:
    print('Display not found. Using Agg backend for matplotlib')
    mpl.use('Agg')
else:
    print(f'Found display : {os.environ["DISPLAY"]}')
import matplotlib.pyplot as plt
import numpy as np
import torch
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

from .criterions import create_criterion
from .data import get_dataloader
from .models import create_model
from .utils.misc import load_config

def main():
    cfg = load_config('configs/dummy.yml')

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)

    print('###### Loading data ######')
    dataloader = get_dataloader(cfg.data)
    print(dataloader)
    print(dataloader.dataset)
    print('###########################')

    print('###### Creating model #####')
    model = create_model(cfg.model)
    print(model)
    print('###########################')

    criterion = create_criterion(cfg.criterion)
    print('###### Creating crit ######')
    print(criterion)
    print('###########################')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('using device', device)

    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

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
                    s = f'e{epoch:4d};  iter{iter:6d};  loss{loss:6.2f};  '
                    s = s + ';  '.join([f'{k}{v:9.1e}' for k,v in metrics.items() if isinstance(v, float)])
                    print(s)

            iter += 1


if __name__ == "__main__":
    main()
