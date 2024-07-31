import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class DualInputTCNModel(nn.Module):
    def __init__(
        self,
        signal_input_size=60,
        metmast_input_size=60,
        num_signal_feature=6,
        num_metmast_feature=4,
        output_size=10,
        num_channels=[32, 32, 64, 64],
        kernel_size=3,
        dropout=0.5,
    ):
        super().__init__()
        self.signal_input_size = signal_input_size
        self.metmast_input_size = metmast_input_size
        self.output_size = output_size

        self.signal_tcn = TemporalConvNet(signal_input_size, num_channels, kernel_size, dropout=dropout)
        self.metmast_tcn = TemporalConvNet(metmast_input_size, num_channels, kernel_size, dropout=dropout)

        self.linear = nn.Linear(2 * num_channels[-1], output_size)

        self.signal_norm = nn.BatchNorm1d(num_signal_feature)
        self.metmast_norm = nn.BatchNorm1d(num_metmast_feature)

        if output_size != 1:
            self.output_norm = nn.BatchNorm1d(output_size)

        self.name = 'DualInputTCN'

    def forward(self, signal, metmast):
        # signal: [B, T_signal, F_signal]
        # metmast: [B, T_metmast, F_metmast]
        signal = signal.permute(0, 2, 1)
        metmast = metmast.permute(0, 2, 1)

        signal = self.signal_norm(signal)
        metmast = self.metmast_norm(metmast)

        signal = signal.permute(0, 2, 1)
        metmast = metmast.permute(0, 2, 1)

        signal_output = self.signal_tcn(signal)
        metmast_output = self.metmast_tcn(metmast)

        output = torch.cat((signal_output[:, :, -1], metmast_output[:, :, -1]), dim=1)
        output = self.linear(output)
        output = output.squeeze()

        if self.output_size != 1:
            output = self.output_norm(output)

        return output