import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from .base import BaseModel

# -----------------------------
# 1. A Simple PyTorch Dataset
# -----------------------------
class TimeSeriesDataset(data.Dataset):
    """
    A simple Dataset to wrap (X, y).
    X: shape (N, seq_len, input_dim)
    y: shape (N,) or (N, output_dim)
    """

    def __init__(self, X, y):
        # Ensure X and y are torch Tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


########################################
# LSTM (Vanilla)
########################################
class LSTMModel(BaseModel, nn.Module):
    """
    A simple LSTM regressor in PyTorch.
    Inherits from BaseModel (abstract) and nn.Module.
    """

    def __init__(self,
                 name='LSTM',
                 input_dim=1,
                 hidden_dim=64,
                 num_layers=1,
                 output_dim=1,
                 device='cpu'):
        BaseModel.__init__(self, name)
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.to(device)

    def forward(self, x):
        # Ensure x is 3D: (batch_size, seq_len, input_dim)
        if len(x.shape) == 2:  # If x is 2D (seq_len, input_dim), add batch dimension
            x = x.unsqueeze(0)  # -> (1, seq_len, input_dim)

        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_len, hidden_dim)
        out = out[:, -1, :]  # last time step -> (batch_size, hidden_dim)

        # Fully connected
        out = self.fc(out)  # (batch_size, output_dim)
        return out

    def fit(self, X, y, epochs=10, batch_size=32, lr=1e-3, verbose=True):
        dataset = TimeSeriesDataset(X, y)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(batch_X)

                # If y is shape (batch,) and outputs is shape (batch,1), match shape.
                if len(batch_y.shape) == 1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(dataloader.dataset)
            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    def predict(self, X):
        self.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self.forward(X)
        return outputs.cpu().numpy()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()


########################################
# GRU
########################################
class GRUModel(BaseModel, nn.Module):
    """
    A simple GRU regressor in PyTorch.
    """

    def __init__(self,
                 name='GRU',
                 input_dim=1,
                 hidden_dim=64,
                 num_layers=1,
                 output_dim=1,
                 device='cpu'):
        BaseModel.__init__(self, name)
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = device

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.to(device)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)

        out, _ = self.gru(x, h0)  # (batch, seq_len, hidden_dim)
        out = out[:, -1, :]       # last time step
        out = self.fc(out)
        return out

    def fit(self, X, y, epochs=10, batch_size=32, lr=1e-3, verbose=True):
        dataset = TimeSeriesDataset(X, y)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(batch_X)

                if len(batch_y.shape) == 1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(dataloader.dataset)
            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    def predict(self, X):
        self.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self.forward(X)
        return outputs.cpu().numpy()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()


########################################
# BiLSTM
########################################
class BiLSTMModel(BaseModel, nn.Module):
    """
    A bidirectional LSTM regressor in PyTorch.
    """

    def __init__(self,
                 name='BiLSTM',
                 input_dim=1,
                 hidden_dim=64,
                 num_layers=1,
                 output_dim=1,
                 device='cpu'):
        BaseModel.__init__(self, name)
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)  # 2 for bidirectional

        self.to(device)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).to(self.device)

        out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, 2*hidden_dim)
        out = out[:, -1, :]             # last time step
        out = self.fc(out)              # (batch, output_dim)
        return out

    def fit(self, X, y, epochs=10, batch_size=32, lr=1e-3, verbose=True):
        dataset = TimeSeriesDataset(X, y)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(batch_X)

                if len(batch_y.shape) == 1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(dataloader.dataset)
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    def predict(self, X):
        self.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self.forward(X)
        return outputs.cpu().numpy()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()


########################################
# Transformer (simple PyTorch example)
########################################
class TransformerModel(nn.Module):
    """
    A simplified Transformer-based regressor in PyTorch.
    """

    def __init__(self,
                 name='Transformer',
                 input_dim=1,
                 d_model=64,
                 nhead=8,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 output_dim=1,
                 device='cpu'):
        nn.Module.__init__(self)

        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, output_dim)

        self.to(device)

    def forward(self, src, tgt):
        """
        src shape: (batch, src_seq_len, input_dim)
        tgt shape: (batch, tgt_seq_len, input_dim)
        """
        src_embedded = self.input_embedding(src)  # (batch, src_seq_len, d_model)
        tgt_embedded = self.input_embedding(tgt)  # (batch, tgt_seq_len, d_model)

        out = self.transformer(src_embedded, tgt_embedded)  # (batch, tgt_seq_len, d_model)
        out = self.fc_out(out)  # => (batch, tgt_seq_len, output_dim)
        return out

    def fit(self, src, tgt, tgt_y, epochs=10, batch_size=32, lr=1e-3, verbose=True):
        """
        A basic fit method for a Transformer example:

        src:  shape (N, src_seq_len, input_dim)
        tgt:  shape (N, tgt_seq_len, input_dim)
        tgt_y: shape (N, tgt_seq_len, output_dim) or (N, tgt_seq_len)

        For simplicity, we combine (src, tgt, tgt_y) into a single Dataset.
        """
        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src, dtype=torch.float32)
        if not isinstance(tgt, torch.Tensor):
            tgt = torch.tensor(tgt, dtype=torch.float32)
        if not isinstance(tgt_y, torch.Tensor):
            tgt_y = torch.tensor(tgt_y, dtype=torch.float32)

        class TransformerDataset(data.Dataset):
            def __init__(self, s, t, ty):
                self.s = s
                self.t = t
                self.ty = ty

            def __len__(self):
                return len(self.s)

            def __getitem__(self, idx):
                return self.s[idx], self.t[idx], self.ty[idx]

        dataset = TransformerDataset(src, tgt, tgt_y)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_src, batch_tgt, batch_tgt_y in dataloader:
                batch_src = batch_src.to(self.device)
                batch_tgt = batch_tgt.to(self.device)
                batch_tgt_y = batch_tgt_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(batch_src, batch_tgt)
                # outputs: (batch, tgt_seq_len, output_dim)

                if len(batch_tgt_y.shape) == 2 and outputs.shape[-1] == 1:
                    batch_tgt_y = batch_tgt_y.unsqueeze(-1)

                loss = criterion(outputs, batch_tgt_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_src.size(0)

            epoch_loss /= len(dataloader.dataset)
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    def predict(self, src, tgt):
        self.eval()
        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src, dtype=torch.float32)
        if not isinstance(tgt, torch.Tensor):
            tgt = torch.tensor(tgt, dtype=torch.float32)

        src = src.to(self.device)
        tgt = tgt.to(self.device)

        with torch.no_grad():
            outputs = self.forward(src, tgt)
        return outputs.cpu().numpy()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
