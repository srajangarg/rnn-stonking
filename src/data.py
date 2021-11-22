"""
Create stonking data loader classes
"""
import datetime
import inspect
import logging
from datetime import date, timedelta
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from pathlib import Path

# A logger for this file
log = logging.getLogger(__name__)

### utilities start
def outlier_removal(minute_df):
    # remove days with any negative prices
    is_negative = minute_df.groupby('trade_date').agg({k: lambda x: x[x <= 0].any() for k in ['close', 'open', 'high', 'low']})
    remove_days = is_negative[is_negative.any(axis=1)].index
    minute_df = minute_df[~minute_df['trade_date'].isin(remove_days)]
    return minute_df

def get_minute_data(connection, contract, start_date, end_date, num_next_futures, remove_outlier_days=True):
    # returns dataframe with columns

    # TO BE USED
    # minute (the relative minute number in a day)
    # day    (the relative day number based on start_date)
    # open_i, low_i, high_i, close_i, volume_i, days_to_expiry_i, where i [0, num_next_futures]

    # FOR BOOKKEEPING
    # ticker_i

    contract_specs = pd.read_sql('select * from futures_contract_spec WHERE ticker LIKE "%s%%";' % contract, con=connection)
    contract_specs['end_date'] = contract_specs[['first_notice_date', 'expiration_date']].min(axis=1)
    contract_specs = contract_specs[(contract_specs.first_notice_date > start_date) & (contract_specs.expiration_date > start_date)].sort_values('expiration_date')

    end_contract_date = end_date + timedelta(days=(num_next_futures+1)*30)
    contract_specs = contract_specs[(contract_specs.first_notice_date < end_contract_date) & (contract_specs.expiration_date < end_contract_date)].sort_values('expiration_date')
    ticker_ids = ', '.join([str(id) for id in contract_specs.id.unique()])

    minute = pd.read_sql(f'select * from futures_contract_one_min where ticker_id in ({ticker_ids}) and trade_date >= "{start_date.strftime("%Y-%m-%d")}" and trade_date <= "{end_date.strftime("%Y-%m-%d")}"', con=connection)
    minute = minute.merge(contract_specs[["id", "ticker", "end_date"]], left_on="ticker_id", right_on="id", how="left")
    minute = minute[minute['trade_date'] < minute['end_date']]
    minute = minute[["open", "high", "low", "close", "volume", "ticker", "end_date", "trade_date", "timestamp"]]

    if remove_outlier_days:
        minute = outlier_removal(minute)

    date_dfs = []
    for day, (trade_date, group) in enumerate(minute.groupby('trade_date')):
        # check sorted by expirys
        sub_dfs = []
        for i, (id, subgroup) in enumerate(list(group.groupby('end_date'))[:num_next_futures]):
            all_minutes = pd.date_range("%s 05:00" % trade_date.strftime("%Y-%m-%d"), "%s 20:59" % trade_date.strftime("%Y-%m-%d"), freq="1min")
            if len(subgroup) < len(all_minutes)/4:
                log.warning(f"less than one-fourth points populated for {trade_date} for future {i}")
            sub_df = subgroup.set_index('timestamp').reindex(all_minutes)
            sub_df["prev_close"]       = sub_df["close"].shift(1)
            sub_df["days_till_expiry"] = np.busday_count(trade_date, sub_df['end_date'].dropna().iloc[0])
            # sub_df = sub_df[["open", "high", "low", "close", "volume",  "end_date"]].add_suffix("_%d" % i)
            sub_df = sub_df[["open", "high", "low", "close", "prev_close", "volume", "ticker", "days_till_expiry"]].add_suffix("_%d" % i)
            sub_dfs.append(sub_df)
        date_df = pd.concat(sub_dfs, axis=1)
        date_df["minute"] = range(len(date_df))
        date_df["day"]    = day
        date_dfs.append(date_df)
    final_df = pd.concat(date_dfs)

    for i in range(num_next_futures):
        final_df['volume_%d' % i]   = final_df['volume_%d' % i].fillna(0)
        final_df['close_%d' % i]    = final_df['close_%d' % i].ffill()
        final_df['prev_close_%d' % i]    = final_df['prev_close_%d' % i].ffill()
        final_df['open_%d' % i]     = final_df['open_%d' % i].fillna(final_df['close_%d' % i])
        final_df['low_%d' % i]      = final_df['low_%d' % i].fillna(final_df['close_%d' % i])
        final_df['high_%d' % i]     = final_df['high_%d' % i].fillna(final_df['close_%d' % i])
        final_df['norm_volume_%d' % i]   = final_df['volume_%d' % i].div(final_df['volume_%d' % i].max(), axis=0)

    return final_df
### utilities end


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
        return create_engine(f'mysql+mysqlconnector://{username}:{password}@{ip}:{port}/{dbname}')


class SingleStonkDataset(StonkBaseDataset):
    """ Loads data for a single stock
    """
    def __init__(self,
            sequence_length:int,
            # db: DictConfig,
            # contract: str,
            # start_date: str,
            # end_date: str,
            # resolution: str,
        ) -> None:
        super().__init__()
        self.sequence_length = sequence_length


        # Concatenate all data in prices/features.
        # idsubsequence_to_idxstart[i] points to index at which i-th subsequence (of length sequence_length) starts
        # self.price = torch.rand(1000000, 1)
        # self.feats = torch.rand(1000000, 5 * num_next_futures)

        # num_minutes_per_day = 960       # Ugly hardcode
        # num_days = len(self.data) // num_minutes_per_day
        # ids = torch.arange(1, num_minutes_per_day - sequence_length - 1, subsequence_shift)
        # ids = ids[:, None] + num_minutes_per_day * torch.arange(num_days)
        # self.idsubsequence_to_idxstart = list(map(int, ids.view(-1)))

        # # Filter out subsequences that have Nan in first prev_close
        # first_row = self.data.iloc[self.idsubsequence_to_idxstart]
        # bad_rows = first_row[[f'prev_close_{i}' for i in range(num_next_futures)]].isnull().any(axis=1)
        # self.idsubsequence_to_idxstart = [id for id,bad in zip(self.idsubsequence_to_idxstart, bad_rows) if not bad]
        # log.info(f'Removed {bad_rows.to_numpy().sum()} subsequences with NaN prev_close')
        # log.info(f'Left with {len(self)} subsequences')
        self.data = pd.read_csv("/Users/garg/code/rnn-stonking/ether_sample.csv")


    def __len__(self):
        """ Returns the total number of training instances in the data
        """
        return 1
        # return len(self.idsubsequence_to_idxstart)

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

        feat_cols = ['Open_15', 'High_15', 'Low_15', 'Close_15', 'VWAP_15', 'Volume_15', 'Count_15', 'Volume_Count', 'Open_60', 'High_60', 'Low_60', 'Close_60', 'VWAP_60', 'Volume_60', 'Count_60', 'Volume_Count', 'Open_240', 'High_240', 'Low_240', 'Close_240', 'VWAP_240', 'Volume_240', 'Count_240', 'Volume_Count']

        seq = self.data.iloc[idx: idx+self.sequence_length+1].copy()
        feats = seq[feat_cols].to_numpy()[:self.sequence_length]
        price = seq[['Close']].to_numpy()


        assert(feats.shape[0] == self.sequence_length)
        assert(price.shape[0] == self.sequence_length+1)
        price = torch.as_tensor(price).float()
        feats = torch.as_tensor(feats).float()
        try:
            assert(torch.isfinite(price).all())
            assert(torch.isfinite(feats).all())
        except AssertionError as e:
            log.error(idx)
            log.error(seq)
            log.error(e, exc_info=True)
            exit(0)

        elem = {
            # 'price': self.price[seq_start_idx: seq_start_idx+self.sequence_length+1],
            # 'feats': self.feats[seq_start_idx: seq_start_idx+self.sequence_length]
            'price': price.float(),
            'feats': feats.float(),
            'datetime': seq['timestamp'].iloc[0],
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
        try:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        except TypeError:
            collated_batch[key] = [elem[key] for elem in batch]
        except Exception as e:
            log.error(e, exc_info=True)
    return collated_batch


def get_dataloader(cfg: DictConfig) -> DataLoader:
    dataset = instantiate(cfg.dataset)
    return DataLoader(dataset, collate_fn=collate_fn, pin_memory=False, **cfg.dataloader)

def get_valdataloader(cfg: DictConfig) -> DataLoader:
    dataset = instantiate(cfg.val_dataset)
    return DataLoader(dataset, collate_fn=collate_fn, pin_memory=False, **cfg.val_dataloader)


@hydra.main(config_path="../configs", config_name="dummy")
def test(cfg : DictConfig) -> None:
    dataloader = get_dataloader(cfg)
    for dd in dataloader:
        print(dd['price'].shape)
        print(dd['feats'].shape)
        # import ipdb; ipdb.set_trace()
        break
#         x=0

if __name__ == "__main__":
    test()
