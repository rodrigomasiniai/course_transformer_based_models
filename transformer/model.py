# Reference:
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
    # https://github.com/huggingface/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/vision_transformer.py#L216

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math


D_MODEL = 512
# BATCH_SIZE = 4096
BATCH_SIZE = 16
# SEQ_LEN = 380
SEQ_LEN = 30
# vOCAB_SIZE = 37_000
VOCAB_SIZE = 1000
D_FF = 2048
n_position=200


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()

        self.pe_mat = self._get_positional_encoding_matrix(d_model=d_model, seq_len=seq_len)

    def _get_positional_encoding_matrix(self, d_model, seq_len):
        # seq_len=SEq_LEN
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


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=D_MODEL, n_heads=8):
        super().__init__()
    
        self.dim = dim # $d_{model}$
        self.n_heads = n_heads # $h$

        self.head_dim = dim // n_heads # $d_{k}$, $d_{v}$

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)

        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.1)
        self.w_o = nn.Linear(dim, dim, bias=False)

    def subsequent_info_mask(self, batch_size, src_seq_len, trg_seq_len):
        """ Prevent positions from attending to subsequent positions. """
        mask = torch.tril(torch.ones(size=(src_seq_len, trg_seq_len)), diagonal=0).bool()
        batched_mask = mask.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, self.n_heads)
        return batched_mask

    def forward(self, q, k, v, masked=False):
        b, l, _ = q.shape
        _, m, _ = k.shape

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(b, l, self.head_dim, self.n_heads)
        k = k.view(b, m, self.head_dim, self.n_heads)
        v = v.view(b, m, self.head_dim, self.n_heads)

        attn_score = torch.einsum("bldn,bmdn->blmn", q, k) # "MatMul" in "Figure 2" in the paper
        if masked:
            subseq_mask = self.subsequent_info_mask(batch_size=b, src_seq_len=l, trg_seq_len=m)
            attn_score.masked_fill_(mask=subseq_mask, value=-math.inf) # "Mask (opt.)"
        attn_score /= (self.head_dim ** 0.5) # "Scale"

        attn_weight = self.softmax(attn_score) # "Softmax"
        attn_weight = self.dropout(attn_weight) # Not in the paper

        x = torch.einsum("blmn,bmdn->bldn", attn_weight, k) # "MatMul"
        x = rearrange(x, "b l d n -> b l (d n)")

        x = self.w_o(x)
        return x


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


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model

        self.self_attn = MultiHeadAttention(dim=d_model, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = PositionwiseFeedForward(d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output = self.self_attn(q=x, k=x, v=x) # "Multi-Head Attention" in "Figure 1" in the paper
        x += attn_output # "Add"
        x = self.norm1(x) # "& Norm"

        ff_output = self.ff(x) # "Feed Forward"
        x += ff_output # "Add"
        x = self.norm2(x) # "& Norm"
        x = self.dropout(x) # Not in the paper
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
        self.enc_stack = nn.ModuleList(
            [EncoderLayer(d_model=d_model, n_heads=n_heads) for _ in range(self.n_layers)]
        )

    def forward(self, x):
        x = self.input(x)
        for enc_layer in self.enc_stack:
            x = enc_layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model

        self.self_attn = MultiHeadAttention(dim=d_model, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.enc_dec_attn = MultiHeadAttention(dim=d_model, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = PositionwiseFeedForward(d_model=d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_output):
        attn_output = self.self_attn(q=x, k=x, v=x) # "Masked Multi-Head Attention" in "Figure 1" in the paper
        x += attn_output # "Add"
        x = self.norm1(x) # "& Norm"

        attn_output = self.enc_dec_attn(q=x, k=enc_output, v=enc_output, masked=True) # "Multi-Head Attention"
        x += attn_output # "Add"
        x = self.norm2(x) # "& Norm"

        ff_output = self.ff(x) # "Feed Forward"
        x += ff_output # "Add"
        x = self.norm3(x) # "& Norm"

        x = self.dropout(x) # Not in the paper
        return x


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, trg_seq_len, n_heads, d_model, n_layers):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size
        self.trg_seq_len = trg_seq_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.input = Input(vocab_size=trg_vocab_size, d_model=d_model, seq_len=trg_seq_len)
        self.dec_stack = nn.ModuleList(
            [DecoderLayer(n_heads=n_heads, d_model=d_model) for _ in range(self.n_layers)]
        )
        self.linear = nn.Linear(d_model, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, enc_output):
        # print(y)
        x = self.input(x)
        for dec_layer in self.dec_stack:
            x = dec_layer(x, enc_output=enc_output)
            # print(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


class Transformer(nn.Module):
    # `n_layers`: $N$
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_seq_len,
        trg_seq_len,
        d_model,
        n_heads=8,
        n_enc_layers=6,
        n_dec_layers=6,
    ):
        super().__init__()

        self.enc = Encoder(
            src_vocab_size=src_vocab_size,
            src_seq_len=src_seq_len,
            n_heads=n_heads,
            d_model=d_model,
            n_layers=n_enc_layers
        )
        self.dec = Decoder(
            trg_vocab_size=trg_vocab_size,
            trg_seq_len=trg_seq_len,
            n_heads=n_heads,
            d_model=d_model,
            n_layers=n_dec_layers
        )

        # "we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation" in section 3.4 in the paper
        # decoder.input.embedding.weight = encoder.input.embedding.weight
        # decoder.linear.weight = decoder.input.embedding.weight

    def forward(self, src_seq, trg_seq):
        enc_output = self.enc(src_seq)
        dec_output = self.dec(trg_seq, enc_output)
        return dec_output


if __name__ == "__main__":
    transformer = Transformer(
        src_vocab_size=1000,
        trg_vocab_size=800,
        src_seq_len=30,
        trg_seq_len=44,
        d_model=D_MODEL
    )

    src_seq = torch.randint(low=0, high=src_vocab_size, size=(BATCH_SIZE, src_seq_len))
    trg_seq = torch.randint(low=0, high=trg_vocab_size, size=(BATCH_SIZE, trg_seq_len))

    output = transformer(src_seq=src_seq, trg_seq=trg_seq)
    output