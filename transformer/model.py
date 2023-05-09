# Reference:
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


D_MODEL = 512
# BATCH_SIZE = 4096
BATCH_SIZE = 16
# SEQ_LEN = 380
SEQ_LEN = 30
# VOCAB_SIZE = 37_000
VOCAB_SIZE = 1000
D_FF = 2048
n_position=200
# """ sinusoid position embedding """
# def get_sinusoid_encoding_table(n_seq, d_hidn):
#     def cal_angle(position, i_hidn):
#         return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
#     def get_posi_angle_vec(position):
#         return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

#     pe_mat = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
#     pe_mat[:, 0::2] = np.sin(pe_mat[:, 0::2])  # even index sin 
#     pe_mat[:, 1::2] = np.cos(pe_mat[:, 1::2])  # odd index cos
#     return pe_mat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()

        self.pe_mat = self._get_positional_encoding_matrix(d_model=d_model, seq_len=seq_len)

    def _get_positional_encoding_matrix(self, d_model, seq_len):
        # seq_len=SEQ_LEN
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


class Input(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, seq_len=seq_len)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embedding(x)
        x *= self.d_model ** 0.5
        x += self.positional_encoding(x)
        x = self.dropout(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) # "MatMul" in "Figure 2" in the paper
        attn_scores /= self.temperature # "Scale"
        if mask is not None:
            attn_scores.masked_fill_(mask=mask, value=-math.inf) # "Mask (opt.)"
        attn_weights = F.softmax(attn_scores, dim=-1) # "Softmax"
        attn_weights = self.dropout(attn_weights) # ?
        x = torch.matmul(attn_weights, V) # MatMul
        return x, attn_weights


class MaskedMultiHeadAttention(nn.Module):
    ### TO DO: 차원 수정!!
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads # $h$
        self.d_model = d_model

        self.d_kv = d_model // n_heads # $d_{k}$, $d_{v}$

        self.W_Q = nn.Linear(d_model, self.d_kv * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_kv * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_kv * n_heads, bias=False)
        self.W_O = nn.Linear(self.d_kv * n_heads, d_model, bias=False)

        self.attn = ScaledDotProductAttention(temperature=self.d_kv ** 0.5)

    def subsequent_info_mask(self, batch_size, seq_len, device):
        """ Prevent positions from attending to subsequent positions. """
        mask = torch.tril(torch.ones(size=(seq_len, seq_len), device=device), diagonal=0).bool()
        batched_mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        return batched_mask

    def forward(self, Q, K, V, masked=False):
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        b, l, _ = Q.shape
        if masked:
            subseq_mask = self.subsequent_info_mask(batch_size=b, seq_len=l, device=Q.device)
            x, attn_weights = self.attn(Q=Q, K=K, V=V, mask=subseq_mask)
        else:
            x, attn_weights = self.attn(Q=Q, K=K, V=V)
        x = self.W_O(x)
        return x, attn_weights
# seq = torch.randn(size=(BATCH_SIZE, SEQ_LEN, D_MODEL))
# multi_head_attn = MaskedMultiHeadAttention(n_heads=8, d_model=D_MODEL)
# a, b = multi_head_attn(seq)
# print(a.shape, b.shape)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ff=D_FF):
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
    def __init__(self, src_vocab_size, src_seq_len, n_heads, d_model, n_layers):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.src_seq_len = src_seq_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.input = Input(vocab_size=src_vocab_size, d_model=d_model, seq_len=src_seq_len)
        # input = Input(vocab_size=src_vocab_size, d_model=d_model, seq_len=src_seq_len)
        # input.embedding.weight.shape

        self.self_attn = MaskedMultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.feed_forward = PositionwiseFeedForward(d_model=d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.input(x)
        for _ in range(self.n_layers):
            attn_output, attn_weights = self.self_attn(Q=x, K=x, V=x) # "Multi-Head Attention" in "Figure 1" in the paper
            x += attn_output # "Add"
            x = self.ln1(x) # "& Norm"

            ff_output = self.feed_forward(x) # "Feed Forward"
            x += ff_output # "Add"
            x = self.ln2(x) # "& Norm"
            x = self.dropout(x) # Not in the paper
        return x, attn_weights


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, trg_seq_len, n_heads, d_model, n_layers):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size
        self.trg_seq_len = trg_seq_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.input = Input(vocab_size=trg_vocab_size, d_model=d_model, seq_len=trg_seq_len)
        # input = Input(vocab_size=trg_vocab_size, d_model=d_model, seq_len=trg_seq_len)
        # input.embedding.weight.shape
        # nn.Linear(d_model, trg_vocab_size).weight.shape

        self.self_attn = MaskedMultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.enc_dec_attn = MaskedMultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.feed_forward = PositionwiseFeedForward(d_model=d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

        self.linear = nn.Linear(d_model, trg_vocab_size)

    def forward(self, x, y):
        x = self.input(x)
        for _ in range(self.n_layers):
            attn_output, attn_weights1 = self.self_attn(Q=x, K=x, V=x, masked=True) # "Masked Multi-Head Attention" in "Figure 1" in the paper
            x += attn_output # "Add"
            x = self.ln1(x) # "& Norm"

            attn_output, attn_weights2 = self.enc_dec_attn(Q=x, K=y, V=y) # "Multi-Head Attention"
            x += attn_output # "Add"
            x = self.ln2(x) # "& Norm"

            ff_output = self.feed_forward(x) # "Feed Forward"
            x += ff_output # "Add"
            x = self.ln3(x) # "& Norm"

            x = self.dropout(x) # Not in the paper

        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        return x, attn_weights1, attn_weights2


class Transformer(nn.Module):
    # `n_layers`: $N$
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_seq_len,
        trg_seq_len,
        d_model,
        n_enc_layers=6,
        n_dec_layers=6,
        n_heads=8
    ):
        super().__init__()
        d_model=D_MODEL
        src_vocab_size=1000
        trg_vocab_size=800
        src_seq_len=30
        trg_seq_len=44
        n_enc_layers=6
        n_dec_layers=6
        n_heads=8

        encoder = Encoder(
            src_vocab_size=src_vocab_size,
            src_seq_len=src_seq_len,
            n_heads=n_heads,
            d_model=d_model,
            n_layers=n_enc_layers
        )
        decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            trg_seq_len=trg_seq_len,
            n_heads=n_heads,
            d_model=d_model,
            n_layers=n_enc_layers
        )

        # "we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation" in section 3.4 in the paper
        decoder.input.embedding.weight = encoder.input.embedding.weight
        decoder.linear.weight = decoder.input.embedding.weight

    def forward(self, src_seq, trg_seq):
        src_seq = torch.randint(low=0, high=src_vocab_size, size=(BATCH_SIZE, src_seq_len))
        trg_seq = torch.randint(low=0, high=trg_vocab_size, size=(BATCH_SIZE, trg_seq_len))

        enc_output, _ = encoder(src_seq)
        dec_output, attn_weights1, attn_weights2 = decoder(trg_seq, enc_output)
        dec_output.shape
        # print(dec_output.shape, attn_weights1.shape, attn_weights2.shape)
        # trg_seq = trg_input(trg_seq)
        print(src_seq.shape)
        print(_.shape)


if __name__ == "__main__":
    input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    # seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    transformer = Transformer(n_heads=6, d_model=d_model)
    transformer(input).shape