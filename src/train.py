# -*- coding: utf-8 -*-
"""
train.py
Training loop for the EnergyLSTM model.

Handles:
- Model initialization and device placement
- Loss function and optimizer setup
- Training and validation loops
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Training history logging
"""
# import libraries
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))

from model import EnergyLSTM
from dataset import (
    split_data, fit_scalers, build_dataloaders,
    SEQUENCE_LENGTH, BATCH_SIZE
)

# hyperparameters
# model architecture
INPUT_SIZE = 13 # features
HIDDEN_SIZE = 128 # neurons per LSTM layer
NUM_LAYERS = 2 # stacked LSTM layers
DROPOUT = 0.2 # dropout rate

# training
LEARNING_RATE = 0.001 # Adam optimizer starting rate
MAX_EPOCHS = 100 # max num of epochs if early stopping doesn't trigger
PATIENCE = 10 # early stoping patience - num of epochs without improvement

# paths
MODEL_PATH = os.path.join('models', 'best_model.pt')

# function declarations
def get_device() -> torch.device:
    """
    Detects and returns the best available device
    """
    #device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    #print(f'Training device: {device}')
    #if device.type == 'cuda':
    #    print(f'GPU: {torch.cuda.get_device_name(0)}')

    device = torch.device('cpu')
    print(f'Training device: {device}')

    # return device
    return device

def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> float:
    """
    Run one full training epoch.

    For each batch:
    1. Move data to device
    2. Zero out gradients from previous batch
    3. Forward pass — get predictions
    4. Compute loss
    5. Backward pass — compute gradients
    6. Update weights

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for sequences, targets in loader:
        # move tensors to correct device
        sequences = sequences.to(device)
        targets = targets.to(device)

        # 1: zero gradients
        optimizer.zero_grad()

        # 2: forward pass
        predictions = model(sequences)

        # 3: compute loss
        loss = criterion(predictions.squeeze(), targets)

        # 4: backward pass - compute gradients
        loss.backward()

        # 5: gradient clipping - prevents exploding gradients 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6 update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model: nn.Module,
             loader,
             criterion: nn.Module,
             device: torch.device) -> float:
    """
    Evaluate model on validation or test data

    model.eval() disables dropout
    torch.no_grad() disables gradient computations

    Returns:
        Average loss for dataset
    """
    # disable dropout
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            predictions = model(sequences)
            loss = criterion(predictions.squeeze(), targets)
            total_loss += loss.item()

    return total_loss / len(loader)

# early stopping class
class EarlyStopping:
    """
    Monitors validation loss and stops training when it stops improving

    Saves best model weights to disk whenever validation loss improves; after training completes, load best weights back so always end up with the best model

    Args: 
        patience:  Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as improvement
        path:      Where to save the best model weights
    """

    # method declarations
    def __init__(self, 
                 patience: int = PATIENCE, 
                 min_delta: float = 0.0001,
                 path: str = MODEL_PATH):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self,
                 val_loss: float,
                 model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            # improvement found - save model and reset the counter
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f'Validation loss improved -> model saved')
        else:
            # no improvement 
            self.counter += 1
            print(f' No improvement({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True

# main training function
def train(df: pd.DataFrame):
    """
    Full training pipeline

    Args:
        df: Feature df from features.py
    """
    # get device
    device = get_device()

    # prep data
    print('\nPreparing data...')
    train_df, val_df, test_df = split_data(df)
    feature_scaler, target_scaler = fit_scalers(train_df)
    train_loader, val_loader, test_loader = build_dataloaders(train_df,
                                                              val_df,
                                                              test_df,
                                                              feature_scaler,
                                                              target_scaler)
    
    # save scalers for inference
    import joblib
    joblib.dump(feature_scaler, os.path.join('models', 'feature_scaler.pkl'))
    joblib.dump(target_scaler, os.path.join('models', 'target_scaler.pkl'))
    print('Scalers saved to models/')

    # init EnergyLSTM model object
    model = EnergyLSTM(input_size=INPUT_SIZE,
                       hidden_size=HIDDEN_SIZE,
                       num_layers=NUM_LAYERS,
                       dropout=DROPOUT,
                       output_size=1).to(device)
    print(f'Parameters: {model.count_parameters():,}')

    # loss function
    criterion = nn.MSELoss() # computes mean squared error between preds and targs

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # learning rate scheduler | reduces learning rate by factor of .5 when val loss plateaus for 5 epochs
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=5)
    
    # init EarlyStopping object
    early_stopping = EarlyStopping(patience=PATIENCE,
                                   path=MODEL_PATH)
    
    # training loop
    print(f'\nStarting training - max {MAX_EPOCHS} epochs, '
          f'early stopping patience={PATIENCE}')
    
    history = {'train_loss': [], 
               'val_loss': [], 
               'lr': []}
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS+1):
        epoch_start = time.time()

        # train
        train_loss = train_one_epoch(model,
                                     train_loader,
                                     optimizer,
                                     criterion,
                                     device)
        
        # validate
        val_loss = evaluate(model,
                            val_loader,
                            criterion,
                            device)
        
        # step scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # log history by appending to lists in history dict
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # print progress
        elapsed = time.time() - epoch_start
        print(f'Epoch {epoch:3d}/{MAX_EPOCHS} | '
              f'Train Loss: {train_loss:.6f} | '
              f'Val Loss: {val_loss:.6f} | '
              f'LR: {current_lr:.6f} | '
              f'Time: {elapsed:.1f}s')
        
        # check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered at epoch {epoch}')
            break

    total_time = time.time() - start_time
    print(f'\nTraining complete in {total_time/60:.1f} minutes')
    print(f'Best validation loss: {early_stopping.best_loss:.6f}')

    # load best model
    print(f'\nLoading best model weights from {MODEL_PATH}')

    # plot training history
    plot_training_history(history)

    return model, feature_scaler, target_scaler, test_loader, history

def plot_training_history(history:dict):
    """
    Plots training and validation loss curves

    Both curves decreasing - model is learning
    Val loss flattening while train loss keeps dropping - overfitting
    Val loss below train loss - unlikely but means more training data needed
    Large gap between train and val - overfitting
    Curves close together - good generalization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='coral')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['lr'], color='green')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('models', 'training_history.png'),dpi=150)
    plt.show()
    print('Training history plot saved to models/training_history.png')

if __name__ == '__main__':
    print('Loading feature matrix....')
    df = pd.read_csv(os.path.join('data', 'processed', 'features.csv'),
                     parse_dates=['datetime_utc'])
    
    print(f'Loaded {len(df)} rows\n')

    # train the model
    model, feature_scaler, target_scaler, test_loader, history = train(df)

    print('\nTraining pipeline complete')
    print(f'Best model saved to: {MODEL_PATH}')