# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init


class DualInputSFMModel(nn.Module):
    def __init__(
        self,
        signal_dim=6,
        metmast_dim=4,
        output_dim=1,
        freq_dim=10,
        hidden_size=64,
        dropout_W=0.0,
        dropout_U=0.0,
    ):
        super().__init__()

        self.signal_dim = signal_dim
        self.metmast_dim = metmast_dim
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Signal input layers
        self.W_i_signal = nn.Parameter(init.xavier_uniform_(torch.empty((self.signal_dim, self.hidden_dim))))
        self.W_ste_signal = nn.Parameter(init.xavier_uniform_(torch.empty(self.signal_dim, self.hidden_dim)))
        self.W_fre_signal = nn.Parameter(init.xavier_uniform_(torch.empty(self.signal_dim, self.freq_dim)))
        self.W_c_signal = nn.Parameter(init.xavier_uniform_(torch.empty(self.signal_dim, self.hidden_dim)))
        self.W_o_signal = nn.Parameter(init.xavier_uniform_(torch.empty(self.signal_dim, self.hidden_dim)))

        # Metmast input layers
        self.W_i_metmast = nn.Parameter(init.xavier_uniform_(torch.empty((self.metmast_dim, self.hidden_dim))))
        self.W_ste_metmast = nn.Parameter(init.xavier_uniform_(torch.empty(self.metmast_dim, self.hidden_dim)))
        self.W_fre_metmast = nn.Parameter(init.xavier_uniform_(torch.empty(self.metmast_dim, self.freq_dim)))
        self.W_c_metmast = nn.Parameter(init.xavier_uniform_(torch.empty(self.metmast_dim, self.hidden_dim)))
        self.W_o_metmast = nn.Parameter(init.xavier_uniform_(torch.empty(self.metmast_dim, self.hidden_dim)))

        # Shared layers
        self.U_i = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_ste = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        self.U_fre = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim)))
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        self.U_c = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_o = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_p = nn.Parameter(init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim)))
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))

        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)
        self.fc_out = nn.Linear(self.output_dim, self.output_dim)

        self.states = []
        self.name = 'DualInputSFM'

    def forward(self, signal, metmast):
        # signal: [N, T_signal, F_signal]
        # metmast: [N, T_metmast, F_metmast]
        signal = signal.permute(0, 2, 1)  # [N, F_signal, T_signal]
        metmast = metmast.permute(0, 2, 1)  # [N, F_metmast, T_metmast]

        time_step = max(signal.shape[2], metmast.shape[2])

        for ts in range(time_step):
            if ts < signal.shape[2]:
                x_signal = signal[:, :, ts]
            else:
                x_signal = torch.zeros((signal.shape[0], self.signal_dim)).to(self.device)

            if ts < metmast.shape[2]:
                x_metmast = metmast[:, :, ts]
            else:
                x_metmast = torch.zeros((metmast.shape[0], self.metmast_dim)).to(self.device)

            if len(self.states) == 0:  # hasn't initialized yet
                self.init_states(x_signal, x_metmast)

            self.get_constants(x_signal, x_metmast)
            p_tm1 = self.states[0]
            h_tm1 = self.states[1]
            S_re_tm1 = self.states[2]
            S_im_tm1 = self.states[3]
            time_tm1 = self.states[4]
            B_U = self.states[5]
            B_W_signal = self.states[6]
            B_W_metmast = self.states[7]
            frequency = self.states[8]

            x_i_signal = torch.matmul(x_signal * B_W_signal[0], self.W_i_signal)
            x_ste_signal = torch.matmul(x_signal * B_W_signal[0], self.W_ste_signal)
            x_fre_signal = torch.matmul(x_signal * B_W_signal[0], self.W_fre_signal)
            x_c_signal = torch.matmul(x_signal * B_W_signal[0], self.W_c_signal)
            x_o_signal = torch.matmul(x_signal * B_W_signal[0], self.W_o_signal)

            x_i_metmast = torch.matmul(x_metmast * B_W_metmast[0], self.W_i_metmast)
            x_ste_metmast = torch.matmul(x_metmast * B_W_metmast[0], self.W_ste_metmast)
            x_fre_metmast = torch.matmul(x_metmast * B_W_metmast[0], self.W_fre_metmast)
            x_c_metmast = torch.matmul(x_metmast * B_W_metmast[0], self.W_c_metmast)
            x_o_metmast = torch.matmul(x_metmast * B_W_metmast[0], self.W_o_metmast)

            x_i = x_i_signal + x_i_metmast + self.b_i
            x_ste = x_ste_signal + x_ste_metmast + self.b_ste
            x_fre = x_fre_signal + x_fre_metmast + self.b_fre
            x_c = x_c_signal + x_c_metmast + self.b_c
            x_o = x_o_signal + x_o_metmast + self.b_o

            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))

            f = ste * fre

            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))

            time = time_tm1 + 1

            omega = torch.tensor(2 * np.pi) * time * frequency

            re = torch.cos(omega)
            im = torch.sin(omega)

            c = torch.reshape(c, (-1, self.hidden_dim, 1))

            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im

            A = torch.square(S_re) + torch.square(S_im)

            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)

            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))

            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p

            self.states = [p, h, S_re, S_im, time, None, None, None, None]

        self.states = []
        return self.fc_out(p).squeeze()

    def init_states(self, x_signal, x_metmast):
        batch_size = x_signal.shape[0]
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim)).to(self.device)
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim)).to(self.device)

        init_state_h = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)

        init_state = torch.zeros_like(init_state_h).to(self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = torch.tensor(0).to(self.device)

        self.states = [
            init_state_p,
            init_state_h,
            init_state_S_re,
            init_state_S_im,
            init_state_time,
            None,
            None,
            None,
            None,
        ]

    def get_constants(self, x_signal, x_metmast):
        constants = []
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(6)])
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(5)])
        constants.append([torch.tensor(1.0).to(self.device) for _ in range(5)])
        array = np.array([float(ii) / self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))

        self.states[5:] = constants