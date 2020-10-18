import os
import shutil
import joblib
import numpy as np
import pandas as pd
import datetime
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ArrayDataset(Dataset):
    def __init__(self, datasets):
        super(ArrayDataset, self).__init__()
        self._length = len(datasets[0])
        for i, data in enumerate(datasets):
            assert len(data) == self._length, \
                "All arrays must have the same length; \
                array[0] has length %d while array[%d] has length %d." \
                % (self._length, i+1, len(data))
        self.datasets = datasets

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return tuple(torch.from_numpy(data[idx]).float() \
                     for data in self.datasets)

class FXDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, source_len, target_len, step):
        super(FXDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.source_len = source_len
        self.target_len = target_len
        self.step = step
        self.scaler = StandardScaler()

    def split_sequence(self, source, target, source_len, target_len, step, target_start_next):
        """ Split sequence with sliding window into
            sequences of context features and target.
            Args:
                source (np.array): Source sequence
                target (np.array): Target sequence
                source_len (int): Length of input sequence.
                target_len (int): Length of target sequence.
                target_start_next (bool): If True, target sequence
                        starts on the next time step of last step of source
                        sequence. If False, target sequence starts at the
                        same time step of source sequence.
            Return:
                X (np.array): sequence of features
                y (np.array): sequence of targets
        """
        assert len(source) == len(target), \
                'Source sequence and target sequence should have the same length.'

        X, y = list(), list()
        if not target_start_next:
            target = np.vstack((np.zeros(target.shape[1], dtype=target.dtype), target))
        for i in range(0, len(source), step):
            # Find the end of this pattern:
            src_end = i + source_len
            tgt_end = src_end + target_len
            # Check if beyond the length of sequence:
            if tgt_end > len(target):
                break
            # Split sequences:
            X.append(source[i:src_end, :])
            y.append(target[src_end:tgt_end, :])
        return np.array(X), np.array(y)

    def prepare_data(self):
        df = pd.read_csv(self.data_dir, parse_dates=['DATE_TIME'])
        self.data = df.iloc[:,1:].values
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        self.src, self.tgt = self.split_sequence(
                self.data,
                self.data,
                self.source_len,
                self.target_len,
                self.step,
                True
        )

    def setup(self, stage=None):
        # Split data into training set and test set :
        test_idx = int(len(self.src) * 0.7)
        src_train, src_test, tgt_train, tgt_test \
            = self.src[:test_idx], self.src[test_idx:], self.tgt[:test_idx], self.tgt[test_idx:]
        # Split training data into train set and validation set:
        src_train, src_val, tgt_train, tgt_val \
            = train_test_split(src_train, tgt_train, test_size=0.25, random_state=1)
        # Prepare datasets
        self.trainset = ArrayDataset([src_train, tgt_train])
        self.valset = ArrayDataset([src_val, tgt_val])
        self.testset = ArrayDataset([src_test, tgt_test])

    def train_dataloader(self):
        self.trainloader = DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                shuffle=True
        )
        return self.trainloader

    def val_dataloader(self):
        self.valloader = DataLoader(
                self.valset,
                batch_size=self.batch_size,
                shuffle=False
        )
        return self.valloader

    def test_dataloader(self):
        self.testloader = DataLoader(
                self.testset,
                batch_size=self.batch_size,
                shuffle=False
        )
        return self.testloader

class Encoder(nn.Module):
    def __init__(self,
                 source_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 bidirectional=False):
        """ Args:
                source_size (int): The expected number of features in the input.
                hidden_size (int): The number of features in the hidden state.
                num_layers (int): Number of recurrent layers.
                dropout (float): dropout probability.
                bidirectional (boolean): whether to use bidirectional model.
        """
        super(Encoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(source_size,
                          hidden_size,
                          num_layers,
                          dropout=dropout if num_layers else 0,
                          bidirectional=bidirectional,
                          batch_first=True)
        self.compress = nn.Linear(num_layers*num_directions, num_layers)

    def forward(self, input, hidden=None):
        """ Args:
                input (batch, seq_len, source_size): Input sequence.
                hidden (num_layers*num_directions, batch, hidden_size): Initial states.
                
            Returns:
                output (batch, seq_len, num_directions*hidden_size): Outputs at every step.
                hidden (num_layers, batch, hidden_size): Final state.
        """
        # Feed source sequences into GRU:
        output, hidden = self.gru(input, hidden)
        # Compress bidirection to one direction for decoder:
        hidden = hidden.permute(1, 2, 0)
        hidden = self.compress(hidden)
        hidden = hidden.permute(2, 0, 1)
        return output, hidden.contiguous()

class Decoder(nn.Module):
    def __init__(self,
                 target_size,
                 hidden_size,
                 num_layers,
                 dropout):
        """ Args:
                target_size (int): The expected number of sequence features.
                hidden_size (int): The number of features in the hidden state.
                num_layers (int): Number of recurrent layers.
                dropout (float): dropout probability.
        """
        super(Decoder, self).__init__()
        self.target_size = target_size
        self.gru = nn.GRU(target_size,
                          hidden_size,
                          num_layers,
                          dropout=dropout if num_layers else 0,
                          batch_first=True)
        self.out = nn.Linear(hidden_size, target_size)

    def forward(self, hidden, pred_len, target=None, teacher_forcing=False):
        """ Args:
                hidden (num_layers, batch, hidden_size): States of the GRU.
                target (batch, seq_len, target_size): Target sequence. If None,
                    the output sequence is generated by feeding the output
                    of the previous timestep (teacher_forcing has to be False).
                teacher_forcing (bool): Whether to use teacher forcing or not.
                
            Returns:
                outputs (batch, seq_len, target_size): Tensor of log-probabilities
                    of words in the target language.
                hidden of shape (1, batch_size, hidden_size): New states of the GRU.
        """
        if target is None:
            assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'

        # Determine constants:
        batch = hidden.shape[1]
        # The starting value to feed to the GRU:
        val = torch.zeros((batch, 1, self.target_size), device=hidden.device)
        if target is not None:
            target = torch.cat([val, target[:, :-1, :]], dim=1)
        # Sequence to record the predicted values:
        outputs = list()
        for i in range(pred_len):
            # Embed the value at ith time step:
            # If teacher_forcing then use the target value at current step
            # Else use the predicted value at previous step:
            val = target[:, i:i+1, :] if teacher_forcing else val
            # Feed the previous value and the hidden to the network:
            output, hidden = self.gru(val, hidden)
            # Predict new output:
            val = self.out(output.relu()).sigmoid()
            # Record the predicted value:
            outputs.append(val)
        # Concatenate predicted values:
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Model & data module:
    fx_dm = FXDataModule(
        data_dir="forex15m/EURUSD-2000-2020-15m.csv",
        batch_size=256,
        source_len=192,
        target_len=32,
        step=4
    )
    fx_dm.prepare_data()
    fx_dm.setup()
    trainloader = fx_dm.train_dataloader()
    # Encoder:
    fx_encoder = Encoder(
        source_size=4,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
        bidirectional=False
    )
    # Decoder:
    fx_decoder = Decoder(
        target_size=4,
        hidden_size=256,
        num_layers=1,
        dropout=0.0
    )
    
    # Training loop:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fx_encoder.to(device)
    fx_decoder.to(device)
    params = list(fx_encoder.parameters()) + list(fx_decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    for epoch in range(3):
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            _, h = fx_encoder(x)
            o, _ = fx_decoder(h, 32, y, True)
            loss = F.mse_loss(o, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("[%3d - %3d] train_loss: %.4f" % (epoch, i, loss.item()))
    
    # Save model:
    if os.path.isdir("./checkpoint"):
        shutil.rmtree("./checkpoint/")
    os.mkdir("./checkpoint/")
    joblib.dump(fx_dm.scaler, "./checkpoint/scaler.save")
    torch.save(fx_encoder.state_dict(), "./checkpoint/fx_encoder.pth")
    torch.save(fx_decoder.state_dict(), "./checkpoint/fx_decoder.pth")

    
    





