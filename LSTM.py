from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn


class DualInputLSTM(nn.Module):
    def __init__(self, signal_input_size=6, metmast_input_size=4, hidden_size=64, output_size=10, num_layers=2, dropout=0.0):
        super().__init__()
        
        self.signal_rnn = nn.LSTM(
            input_size=signal_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        
        self.metmast_rnn = nn.LSTM(
            input_size=metmast_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        
        self.fc = nn.Linear(2 * hidden_size, hidden_size)
        self.attention_pool = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.signal_norm = nn.BatchNorm1d(signal_input_size)
        self.metmast_norm = nn.BatchNorm1d(metmast_input_size)
        
        if output_size != 1:
            self.output_norm = nn.BatchNorm1d(output_size)
        
        self.signal_input_size = signal_input_size
        self.metmast_input_size = metmast_input_size
        self.output_size = output_size
        self.name = 'DualInputLSTM'
    
    def forward(self, signal, metmast):
        # signal: [B, T_signal, F_signal]
        # metmast: [B, T_metmast, F_metmast]
        signal = signal.permute(0, 2, 1)  # [B, F_signal, T_signal]
        metmast = metmast.permute(0, 2, 1)  # [B, F_metmast, T_metmast]
        
        signal = self.signal_norm(signal)  # [B, F_signal, T_signal]
        metmast = self.metmast_norm(metmast)  # [B, F_metmast, T_metmast]
        
        signal = signal.permute(0, 2, 1)  # [B, T_signal, F_signal]
        metmast = metmast.permute(0, 2, 1)  # [B, T_metmast, F_metmast]
        
        signal_out, _ = self.signal_rnn(signal)  # [B, T_signal, hidden_size]
        metmast_out, _ = self.metmast_rnn(metmast)  # [B, T_metmast, hidden_size]
        
        out = torch.cat((signal_out[:, -1, :], metmast_out[:, -1, :]), dim=1)  # [B, 2 * hidden_size]
        out = self.fc(out)  # [B, hidden_size]
        
        attention_weights = torch.softmax(self.attention_pool(out), dim=1)  # [B, 1]
        pooled_out = out * attention_weights  # [B, hidden_size]        
        out = self.fc_out(pooled_out).squeeze()  # [B, output_size]
        
        if self.output_size != 1:
            out = self.output_norm(out)  # [B, output_size]
        
        out = torch.sigmoid(out)  # [B, output_size]
        
        return out