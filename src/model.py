# -*- coding: utf-8 -*-
"""
model.py
Defines the LSTM neural network architecture for energy demand forecasting.

Architecture:
    Input → Stacked LSTM (2 layers) → Dropout → Linear → Output
"""
# import libraries
import torch
import torch.nn as nn

# EnergyLSTM class declaration
class EnergyLSTM(nn.Module):
    """
    Stacked LSTM model for univariate/multivariate time series forecasting.

    Inherits from nn.Module
    
    __init__: define the layers
    forward:  define how data flows through those layers

    Args:
        input_size:  Number of features per time step (13 in our case)
        hidden_size: Number of neurons in each LSTM layer (128)
        num_layers:  Number of stacked LSTM layers (2)
        dropout:     Fraction of neurons dropped during training (0.2)
        output_size: Number of output values (1 — next hour demand)
    """
    # class method declarations
    def __init__(self,
                 input_size: int=13,
                 hidden_size: int=128,
                 num_layers: int=2,
                 dropout: float=0.2,
                 output_size: int=1):
        """"""
        # init pytorch's internal bookkeeping
        super(EnergyLSTM, self).__init__()

        # store hyperparams that are needed later for hidden state init
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        
        # dropout layer - applied after LSTM/before linear layer | randomly zeros 20% of neurons during training and auto disabled during evolution
        self.dropout = nn.Dropout(p=dropout)
        
        # fully connected output layer - maps hidden_size to output_size | produces single demand prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of how data flows through the model

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               e.g. (32, 168, 13)

        Returns:
            Predictions tensor of shape (batch_size, 1)
        """
        # init hidden state | shape(num_layers, batch_size, hidden_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device) # initial hidden state (short-term memory) | x.device ensures tensors are on same device (CPU/GPU)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device) # initial cell state (long-term memory)

        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # extract last time step - lstm_out contains outputs for all 168 steps, only want the last one
        last_hidden = lstm_out[:, -1, :]

        # dropout
        out = self.dropout(last_hidden)

        # linear output layer
        out = self.fc(out)

        # return the output
        return out
    
    def count_parameters(self) -> int:
        """Counts total number of trainable paramaters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == '__main__':
    print('Building EnergyLSTM model...')

    # create object of EnergyLSTM class
    model = EnergyLSTM(input_size=13,
                        hidden_size=128,
                        num_layers=2,
                        dropout=0.2,
                        output_size=1)
    
    print(model)
    print(f'\nTotal trainable params: {model.count_parameters():,}')

    # forward pass test
    print('\nRunning forward pass with dummy batch...')
    dummy_input = torch.randn(32, 168, 13)
    output = model(dummy_input)

    print(f'Input shape:  {dummy_input.shape}')
    print(f'Output shape: {output.shape}')
    print(f'\nExpected output shape: torch.Size([32, 1])')

    # ── DEVICE CHECK ──────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice available: {device}')
    if device.type == 'cuda':
        print('GPU detected — training will be significantly faster')
    else:
        print('No GPU detected — training on CPU')
        print('This is fine for our dataset size, expect ~10-20 min training time')
        