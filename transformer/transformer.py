# Reference: https://github.com/jadore801120/attention-is-all-you-need-pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


""" sinusoid position embedding """
def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    pe_mat = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    pe_mat[:, 0::2] = np.sin(pe_mat[:, 0::2])  # even index sin 
    pe_mat[:, 1::2] = np.cos(pe_mat[:, 1::2])  # odd index cos
    return pe_mat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()

        self.pe_mat = self._get_positional_encoding_matrix(d_model=d_model, seq_len=seq_len)

    def _get_positional_encoding_matrix(self, d_model, seq_len):
        # seq_len=380
        a, b = np.meshgrid(np.arange(d_model), np.arange(seq_len))
        pe_mat = b / 10000 ** (2 * (a // 2) / d_model)
        pe_mat[:, 0:: 2] = np.sin(pe_mat[:, 0:: 2])
        pe_mat[:, 1:: 2] = np.cos(pe_mat[:, 1:: 2])
        return pe_mat
    
    def forward(self, x):
        b = x.shape[0]
        return nn.Parameter(
            torch.from_numpy(self.pe_mat).unsqueeze(0).repeat(b, 1, 1)
        )


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.temperature

        # if mask is not None:
        #     attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        x = torch.matmul(attn_weights, V)
        return x, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        n_heads = 8 # `h`
        d_k = d_v = d_model // n_heads
        
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.scaled_dot_product_attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, Q, K, V, mask=None):
        heads = [
            self.scaled_dot_product_attention(Q=self.W_Q(Q), K=self.W_K(K), V=self.W_V(V))
            for _ in range(self.n_heads)
        ]
        x = torch.cat(heads, axis=2)
        x = self.W_O(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.w1(x)
        x = self.relu(x)
        x = self.w2(x)
        return x


class Encoder(nn.Module):
    # `n_layers`: $N$
    def __init__(self, n_heads, d_model, n_layers=6):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.multi_head_attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.positional_feed_forward = PositionwiseFeedForward(d_model=d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for _ in range(self.n_layers):
            temp_x, attn_weights = self.multi_head_attn(Q=x, K=x, V=x)
            x += temp_x
            x = self.ln1(x)
            x += self.positional_feed_forward(x)
            x = self.ln2(x)
            x = self.dropout(x)
        return x, attn_weights


class Decoder(nn.Module):
    def __init__(self, n_heads, d_model, n_layers=6):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.multi_head_attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.positional_feed_forward = PositionwiseFeedForward(d_model=d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, self_attn_mask):
        all_attn_weights = list()
        for _ in range(self.n_layers):
            temp_x, attn_weights = self.multi_head_attn(Q=x, K=x, V=x)
            x += temp_x
            x = self.ln1(x)
            x += self.positional_feed_forward(x)
            x = self.ln2(x)
            x = self.dropout(x)

            all_attn_weights.append(attn_weights)
        return x, all_attn_weights


class Transformer(nn.Module):
    # `n_layers`: $N$
    def __init__(self, n_heads, d_model, seq_len=380, n_layers=6):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, seq_len=seq_len)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embedding(x)
        x += self.positional_encoding(x)
        x = dropout(x)
        return x
input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
transformer = Transformer(n_heads=6, d_model=d_model)
transformer(input).shape






batch_size = 4
vocab_size = 1_000
seq_len = 380
d_model=512
d_word_vec=512
d_inner=2048
n_layers=6
d_k=64
d_v=64
dropout=0.1
n_position=200
trg_emb_prj_weight_sharing=True
emb_src_trg_weight_sharing=True


