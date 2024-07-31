import torch
import torch.nn as nn
import torch.nn.functional as F

from attn import AttentionLayer, AnomalyAttention
from embed import DataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list

class DualInputEncoder(nn.Module):
    def __init__(self, win_size, signal_embedding, metmast_embedding, d_model, n_heads, e_layers, d_ff, dropout, activation):
        super(DualInputEncoder, self).__init__()
        self.signal_embedding = signal_embedding
        self.metmast_embedding = metmast_embedding
        self.fc = nn.Linear(2 * win_size, win_size)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=True),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, signal, metmast):
        # signal: [B, T_signal, F_signal]
        # metmast: [B, T_metmast, F_metmast]
        signal_enc = self.signal_embedding(signal)  # [B, T_signal, d_model]
        metmast_enc = self.metmast_embedding(metmast)  # [B, T_metmast, d_model]
        enc_out = torch.cat((signal_enc, metmast_enc), dim=1)  # [B, T_signal + T_metmast, d_model]
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.fc(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out, _, _, _ = self.encoder(enc_out)  # [B, T_signal + T_metmast, d_model]
        return enc_out

class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, signal_enc_in, metmast_enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu'):
        super(AnomalyTransformer, self).__init__()

        # Encoding
        self.signal_embedding = DataEmbedding(signal_enc_in, d_model, dropout)
        self.metmast_embedding = DataEmbedding(metmast_enc_in, d_model, dropout)

        # Dual Input Encoder
        self.dual_input_encoder = DualInputEncoder(win_size, self.signal_embedding, self.metmast_embedding, d_model, n_heads, e_layers, d_ff, dropout, activation)

        # Attention-based pooling
        self.attention_pool = nn.Linear(d_model, 1)

        # Output projection
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, signal, metmast):
        # signal: [B, T_signal, F_signal]
        # metmast: [B, T_metmast, F_metmast]
        enc_out = self.dual_input_encoder(signal, metmast)  # [B, T_signal + T_metmast, d_model]

        # Attention-based pooling
        attention_weights = torch.softmax(self.attention_pool(enc_out), dim=1)  # [B, T_signal + T_metmast, 1]
        pooled_out = torch.sum(enc_out * attention_weights, dim=1)  # [B, d_model]

        # Output projection
        output = self.projection(pooled_out)  # [B, c_out]
        output = torch.sigmoid(output)  # [B, c_out]

        return output
