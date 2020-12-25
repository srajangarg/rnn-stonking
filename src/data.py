"""
Create stonking data loader classes
"""
import inspect
from typing import Dict, List
import hydra

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

class StonkBaseDataset(Dataset):
    """ Base class for all stonking datasets,
        can put common processing code here
    """
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SingleStonkDataset(StonkBaseDataset):
    """ Loads data for a single stock
    """
    def __init__(self, feature_size:int, sequence_length:int) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.sequence_length = sequence_length

        self.price = torch.rand(len(self), self.sequence_length+1, 1)
        self.feats = torch.rand(len(self), self.sequence_length, self.feature_size)

    def __len__(self):
        """ Returns the total number of training instances in the data
        """
        # @srajan-garg: TODO
        return 10000

    def __getitem__(self, idx: int):
        """ Args:
                idx: training index \in [0, self.len)

            Returns a dict with 2 keys:
                price:  A single normalized tensor of size (sequence_length+1, 1)
                        specifying prices at which trades are executed. price[t] is
                        the trade-price at which the positions predicted using feats[t]
                        are obtained. We need an extra element (at sequence_length+1)
                        for the price at which final positions are sold.

                        price[t] would typically be something between low/high at t+1.

                feats:  A single normalized tensor of size (sequence_length, feature_size)
                        containing open/close/low/high/volume/pos etc
        """
        # @srajan-garg: TODO
        elem = {
            'price': self.price[idx],
            'feats': self.feats[idx]
        }
        return elem


def collate_fn(batch: List[Dict[str,torch.Tensor]]) -> Dict[str,torch.Tensor]:
    ''' Globe data collater.
        Assumes each instance is a dict.
        Applies default collation rules for each field.
        Args:
            batch: List of loaded elements via Dataset.__getitem__
    '''
    assert len(batch) > 0
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = default_collate([elem[key] for elem in batch])
    return collated_batch


def get_dataloader(cfg: DictConfig) -> DataLoader:
    dataset = instantiate(cfg.dataset)
    return DataLoader(dataset, collate_fn=collate_fn, **cfg.dataloader)

@hydra.main(config_path="../configs", config_name="dummy")
def test(cfg : DictConfig) -> None:
    dataloader = get_dataloader(cfg.data)
    for dd in dataloader:
        print(dd['price'].shape)
        print(dd['feats'].shape)
        import ipdb; ipdb.set_trace()
        x=0

if __name__ == "__main__":
    test()
