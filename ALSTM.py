import torch
import torch.nn as nn


class DualInputALSTM(nn.Module):
    def __init__(self, signal_input_size=6, metmast_input_size=4, hidden_size=64, output_size=10, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.name = 'DualInputALSTM'
        self.hid_size = hidden_size
        self.signal_input_size = signal_input_size
        self.metmast_input_size = metmast_input_size
        self.output_size = output_size
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()
    
    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        
        self.signal_net = nn.Sequential()
        self.signal_net.add_module("fc_in", nn.Linear(in_features=self.signal_input_size, out_features=self.hid_size))
        self.signal_net.add_module("act", nn.Tanh())
        
        self.metmast_net = nn.Sequential()
        self.metmast_net.add_module("fc_in", nn.Linear(in_features=self.metmast_input_size, out_features=self.hid_size))
        self.metmast_net.add_module("act", nn.Tanh())
        
        self.signal_rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        
        self.metmast_rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        
        self.fc = nn.Linear(2 * self.hid_size, self.hid_size)
        self.fc_out = nn.Linear(in_features=self.hid_size, out_features=self.output_size)
        
        self.signal_norm = nn.BatchNorm1d(self.signal_input_size)
        self.metmast_norm = nn.BatchNorm1d(self.metmast_input_size)
        
        if self.output_size != 1:
            self.output_norm = nn.BatchNorm1d(self.output_size)
        
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in", nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out", nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))
    
    def forward(self, signal, metmast):
        # signal: [B, T_signal, F_signal]
        # metmast: [B, T_metmast, F_metmast]
        signal = signal.permute(0, 2, 1)  # [B, F_signal, T_signal]
        metmast = metmast.permute(0, 2, 1)  # [B, F_metmast, T_metmast]
        
        signal_norm = self.signal_norm(signal)  # [B, F_signal, T_signal]
        metmast_norm = self.metmast_norm(metmast)  # [B, F_metmast, T_metmast]
        
        signal_norm = signal_norm.permute(0, 2, 1)  # [B, T_signal, F_signal]
        metmast_norm = metmast_norm.permute(0, 2, 1)  # [B, T_metmast, F_metmast]
        
        signal_rnn_out, _ = self.signal_rnn(self.signal_net(signal_norm))  # [B, T_signal, hid_size]
        metmast_rnn_out, _ = self.metmast_rnn(self.metmast_net(metmast_norm))  # [B, T_metmast, hid_size]
        
        signal_att_score = self.att_net(signal_rnn_out)  # [B, T_signal, 1]
        metmast_att_score = self.att_net(metmast_rnn_out)  # [B, T_metmast, 1]
        
        signal_out_att = torch.mul(signal_rnn_out, signal_att_score)  # [B, T_signal, hid_size]
        metmast_out_att = torch.mul(metmast_rnn_out, metmast_att_score)  # [B, T_metmast, hid_size]
        
        signal_out_att = torch.sum(signal_out_att, dim=1)  # [B, hid_size]
        metmast_out_att = torch.sum(metmast_out_att, dim=1)  # [B, hid_size]
        
        out = torch.cat((signal_out_att, metmast_out_att), dim=1)  # [B, 2 * hid_size]
        out = self.fc(out)  # [B, hid_size]
        out = self.fc_out(out)  # [B, output_size]
        out = out.squeeze()  # [B]
        
        if self.output_size != 1:
            out = self.output_norm(out)  # [B, output_size]
        
        out = torch.sigmoid(out)  # [B, output_size]
        
        return out