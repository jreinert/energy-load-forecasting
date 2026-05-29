# -*- coding: utf-8 -*-
"""
dataset.py
Handles all data preparation for the LSTM model:
- Train/validation/test splitting
- Normalization
- Sequence window creation
- PyTorch Dataset and DataLoader construction
"""

# import libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# set constant values
SEQUENCE_LENGTH = 168 # 1 week of hourly data
BATCH_SIZE = 32 # number of sequences per training step
TRAIN_FRAC = 0.70 # 70% of data for training
VAL_FRAC = 0.15 # 15% for validation; remaining 15% is test

# features fed into the model
FEATURE_COLS = ['temp_c',
                'hour_sin',
                'hour_cos',
                'dow_sin',
                'dow_cos',
                'month_sin',
                'month_cos',
                'lag_24',
                'lag_48',
                'lag_168',
                'rolling_mean_24',
                'rolling_mean_168',
                'is_holiday']

# target 
TARGET_COL = 'demand_mw'

# function declarations
# train/validate/test split
def split_data(df: pd.DataFrame):
    """
    Splits data chronologically into train, validation, and test sets

    Args:
        df: Full feature DataFrame sorted by datetime_utc

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n* (TRAIN_FRAC + VAL_FRAC))

    # create training, validation, testing dfs using slices of indicies
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    print(f'Train:      {len(train_df)} rows '
          f"({train_df['datetime_utc'].min().date()} to "
          f"{train_df['datetime_utc'].max().date()})")
    print(f'Validation: {len(val_df)} rows '
          f"({val_df['datetime_utc'].min().date()} to "
          f"{val_df['datetime_utc'].max().date()})")
    print(f'Test:       {len(test_df)} rows '
          f"({test_df['datetime_utc'].min().date()} to "
          f"{test_df['datetime_utc'].max().date()})")

    # return dfs
    return train_df, val_df, test_df

# normalization of data
def fit_scalers(train_df: pd.DataFrame):
    """
    Fit StandardScaler on only the training data; transforms each feature to mean = 0, std = 1 and intended to put all features on a comparable scale so no single feature dominates the gradient updates
    
    Returns:
        Tuple of (feature_scaler, target_scaler)
    """
    # create standard scaler object for the features and the target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # fit feature and target datasets to the scalers
    feature_scaler.fit(train_df[FEATURE_COLS])
    target_scaler.fit(train_df[[TARGET_COL]])
    
    # return normalized dfs
    return feature_scaler, target_scaler

def apply_scalers(df: pd.DataFrame,
                  feature_scaler: StandardScaler,
                  target_scaler: StandardScaler) -> pd.DataFrame:
    """
    Applies pre-fitted scalars to a df and returns df with scaled values
    """
    # create copy of df passed as param
    scaled = df.copy()
    scaled[FEATURE_COLS] = feature_scaler.transform(df[FEATURE_COLS])
    scaled[TARGET_COL] = target_scaler.transform(df[[TARGET_COL]])

    # return df
    return scaled

# pytorch dataset class and method declarations
class EnergyDataset(Dataset):
    """
    PyTorch Dataset for energy demand forecasting

    Implements sliding window approach where each sample is a sequence of SEQUENCE_LENGTH consecutive hours of features paired with the demand value at the next hour

    __init__: stores the data
    __len__: return the number of samples
    __getitem__: return one sample by index; DataLoader calls repeatedly to build batches

    Args:
        df: Scaled df with features and target
        sequence_length: # of hrs in each input sequence
    """

    def __init__(self,
                 df: pd.DataFrame,
                 sequence_length: int = SEQUENCE_LENGTH):
        """"""
        # set self.sequence_length
        self.sequence_length = sequence_length

        # convert to np.array for fast indexing
        # shape: (n_rows, n_features)
        self.features = df[FEATURE_COLS].values.astype(np.float32)

        # shape: (n_rows, 1)
        self.target = df[TARGET_COL].values.astype(np.float32)

    def __len__(self):
        """
        # valid sequences in the dataset
        Last valid sequence starts at index (n - sequence_length - 1)
        """
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        """
        Return 1 sequence: target pair

        Sequence shape: (sequence_length, n_features)
        
        target shape: scalar

        Sequence covers hrs [idx, idx + sequence_length]
        Target is the demand at hour [idx + sequence_length]; the very next horu after the sequence ends
        """
        sequence = self.features[idx: idx + self.sequence_length]
        target = self.target[idx + self.sequence_length]

        # convert to pytorch tensors and return tensors
        return torch.tensor(sequence), torch.tensor(target)
    
# dataloaders
def build_dataloaders(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      feature_scaler: StandardScaler,
                      target_scaler: StandardScaler,
                      batch_size: int = BATCH_SIZE):
    """
    Builds PyTorch DataLoaders for train, validation, and test sets

    DataLoader wraps a dataset and handles batching, shuffling, and parallelization of data loading

    Args: 
        batch_size: # of sequences per batch 

    Returns: tuple of (train_loader, val_loader, test_loader)
    """

    # scale all 3 splits using training scalers
    train_scaled = apply_scalers(train_df, feature_scaler, target_scaler)
    val_scaled = apply_scalers(val_df, feature_scaler, target_scaler)
    test_scaled = apply_scalers(test_df, feature_scaler, target_scaler)

    # build dataset objects
    train_dataset = EnergyDataset(train_scaled)
    val_dataset = EnergyDataset(val_scaled)
    test_dataset = EnergyDataset(test_scaled)

    print(f'\nDataset sizes:')
    print(f'Train batches: {len(train_dataset)} sequences')
    print(f'Validation batches: {len(val_dataset)} sequences')
    print(f'Test batches: {len(test_dataset)} sequences')

    # build dataloader objects
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True, # randomize order each epoch
                              num_workers=0) # 0 = load on main thread
    
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0) # keep order for evaluation
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    
    # return loaders
    return train_loader, val_loader, test_loader

# main
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_pipeline import load_raw_data

    # load data    
    print('Loading feature matrix...')
    df = pd.read_csv(os.path.join('data', 'processed', 'features.csv'),
                     parse_dates=['datetime_utc'])
    print(f'Loaded {len(df)} rows, {len(df.columns)} columns\n')

    # split df into train, validate, test sets
    train_df, val_df, test_df = split_data(df)

    # fit scalers on training data
    print('\nFitting scalers on training data...')
    feature_scaler, target_scaler = fit_scalers(train_df)

    # build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, feature_scaler, target_scaler)

    # inspect 1 batch to confirm shapes 
    print('\nInspecting one training batch...')
    sequences, targets = next(iter(train_loader))
    print(f'Sequence shape: {sequences.shape}')
    print(f'Target shape: {targets.shape}')
    print(f'\nExpected sequence shape: '
          f'(batch_size={BATCH_SIZE}, '
          f'sequence_length={SEQUENCE_LENGTH}, '
          f'n_features={len(FEATURE_COLS)})')