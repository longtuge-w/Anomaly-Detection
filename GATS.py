import torch
import torch.nn as nn


class DualInputGATModel(nn.Module):
    def __init__(self, signal_dim=6, metmast_dim=4, hidden_size=64, num_layers=2, output_size=1, dropout=0.0, base_model="GRU"):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.signal_dim = signal_dim
        self.metmast_dim = metmast_dim
        
        if base_model == "GRU":
            self.signal_rnn = nn.GRU(
                input_size=signal_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.metmast_rnn = nn.GRU(
                input_size=metmast_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.signal_rnn = nn.LSTM(
                input_size=signal_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.metmast_rnn = nn.LSTM(
                input_size=metmast_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        
        self.signal_transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.metmast_transformation = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        
        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.name = 'DualInputGAT'

    
    def cal_attention(self, x, y):
        x = self.signal_transformation(x)
        y = self.metmast_transformation(y)
        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(y.expand(sample_num, sample_num, dim), 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight
    
    
    def forward(self, signal, metmast):
        # signal: [N, T_signal, F_signal]
        # metmast: [N, T_metmast, F_metmast]
        signal_out, _ = self.signal_rnn(signal)
        metmast_out, _ = self.metmast_rnn(metmast)
        
        signal_hidden = signal_out[:, -1, :]
        metmast_hidden = metmast_out[:, -1, :]
        
        att_weight = self.cal_attention(signal_hidden, metmast_hidden)
        attended_signal = att_weight.mm(signal_hidden)
        attended_metmast = att_weight.mm(metmast_hidden)
        
        hidden = torch.cat((attended_signal, attended_metmast), dim=1)
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        
        return self.fc_out(hidden).squeeze()