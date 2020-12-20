"""
Create stonking data loader classes
"""
import inspect
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from .utils.misc import get_default_kwargs


class StonkBaseDataset(Dataset):
    """ Base class for all stonking datasets,
        can put common processing code here
    """
    def __init__(self, cfg) -> None:
        super().__init__()

    def __getitem__(self, idx: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SingleStonkDataset(StonkBaseDataset):
    """ Loads data for a single stock
    """
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.feature_size = cfg.feature_size
        self.sequence_length = cfg.sequence_length

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
            'price': torch.rand(self.sequence_length+1, 1),
            'feats': torch.rand(self.sequence_length, self.feature_size)
        }
        return elem


def get_dataset_class_by_name(name):
    """ Searches current file for Dataset classes with matching (case-insensitive) name
    """
    all_classes = {k.lower():v for k,v in globals().items() if (
                        inspect.isclass(v) and
                        (Dataset in inspect.getmro(v))
                    )}
    return all_classes[name.lower()]


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


def get_dataloader(cfg) -> DataLoader:
    dataset_class = get_dataset_class_by_name(cfg.dataset.name)
    kwargs = get_default_kwargs(DataLoader.__init__)
    kwargs.update({k:v for k,v in cfg.items() if k in kwargs})
    kwargs.update(collate_fn=collate_fn)
    return DataLoader(dataset_class(cfg.dataset), **kwargs)

if __name__ == "__main__":
    from .utils.misc import load_config
    cfg = load_config('configs/dummy.yml')
    dataloader = get_dataloader(cfg.data)

    for dd in dataloader:
        print(dd['price'].shape)
        print(dd['feats'].shape)
        import ipdb; ipdb.set_trace()
        x=0
