"""
Create stonking data loader classes
"""
import inspect
from typing import Dict, List
import hydra
import datetime

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

    @staticmethod
    def connect_to_db(ip, port, username, password, dbname):
        # TODO
        return None


class SingleStonkDataset(StonkBaseDataset):
    """ Loads data for a single stock
    """
    def __init__(self,
            sequence_length:int,
            db: DictConfig,
            ticker: str,
            start_date: str,
            end_date: str,
            resolution: str,
            num_next_futures: int,
        ) -> None:
        super().__init__()
        self.sequence_length = sequence_length

        # Create sql connection db.{ip,port,username,password,dbname}
        conn = self.connect_to_db(**db)

        # Extract and filter data from sql usind ticker/start/end dates
        start_date = datetime.date(*list(map(int,start_date.split('-'))))
        end_date = datetime.date(*list(map(int,end_date.split('-'))))
        assert(resolution == '1min')
        ...

        # Concatenate all data in prices/features.
        # idsubsequence_to_idxstart[i] points to index at which i-th subsequence (of length sequence_length) starts
        self.price = torch.rand(1000000, 1)
        self.feats = torch.rand(1000000, 5 * num_next_futures)
        self.idsubsequence_to_idxstart = [i for i in range(10000)]    # TODO

    def __len__(self):
        """ Returns the total number of training instances in the data
        """
        return len(self.idsubsequence_to_idxstart)

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
        # @srajan-garg: TODO might have to normalize prices/features
        seq_start_idx = self.idsubsequence_to_idxstart[idx]
        elem = {
            'price': self.price[seq_start_idx: seq_start_idx+self.sequence_length+1],
            'feats': self.feats[seq_start_idx: seq_start_idx+self.sequence_length]
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
    dataloader = get_dataloader(cfg)
    for dd in dataloader:
        print(dd['price'].shape)
        print(dd['feats'].shape)
        import ipdb; ipdb.set_trace()
        x=0

if __name__ == "__main__":
    test()
