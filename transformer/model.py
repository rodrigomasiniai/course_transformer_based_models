# Reference:
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
    # https://github.com/huggingface/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/vision_transformer.py#L216
    # https://wikidocs.net/31379

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Literal

D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
DROP_PROB = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int=5000) -> None:
        super().__init__()

        self.dim = dim

        pos = torch.arange(max_len).unsqueeze(1) # $pos$
        i = torch.arange(dim // 2).unsqueeze(0) # $i$
        angle = pos / (10_000 ** (2 * i / dim)) # $ \sin(\text{pos} / 10000^{2 * i  / d_{\text{model}}})$

        self.pe_mat = torch.zeros(size=(max_len, dim))
        self.pe_mat[:, 0:: 2] = torch.sin(angle) # $text{PE}_(\text{pos}, 2i)$
        self.pe_mat[:, 1:: 2] = torch.cos(angle) # $text{PE}_(\text{pos}, 2i + 1)$

        self.register_buffer("pos_enc_mat", self.pe_mat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, l, _ = x.shape
        x += self.pe_mat.unsqueeze(0)[:, : l, :]
        return x


class Input(nn.Module):
    def __init__(self, vocab_size, dim, pad_idx=0, drop_prob=DROP_PROB):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.pad_idx = pad_idx

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(dim=dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.embed(x)
        x *= self.dim ** 0.5 # "In the embedding layers
            # we multiply those weights by $\sqrt{d_{text{model}}}$."
            # in section 3.4 of the paper.
        x = self.pos_enc(x)
        x = self.dropout(x) # Not in the paper
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, drop_prob=DROP_PROB):
        super().__init__()
    
        self.dim = dim # $d_{model}$
        self.n_heads = n_heads # $h$

        self.head_dim = dim // n_heads # $d_{k}$, $d_{v}$

        self.q_proj = nn.Linear(dim, dim, bias=False) # $W^{Q}_{i}$
        self.k_proj = nn.Linear(dim, dim, bias=False) # $W^{K}_{i}$
        self.v_proj = nn.Linear(dim, dim, bias=False) # $W^{V}_{i}$

        # self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(dim, dim, bias=False) # $W^{O}$

    def forward(self, q, k, v, mask=None):
        b, l, _ = q.shape
        _, m, _ = k.shape

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q = q.view(b, l, self.head_dim, self.n_heads)
        k = k.view(b, m, self.head_dim, self.n_heads)
        v = v.view(b, m, self.head_dim, self.n_heads)

        attn_score = torch.einsum("bldn,bmdn->blmn", q, k) # "MatMul" in "Figure 2" of the paper
        if mask is not None:
            attn_score.masked_fill_(mask=mask, value=-1e9) # "Mask (opt.)"
        attn_score /= (self.head_dim ** 0.5) # "Scale"

        # attn_weight = self.softmax(attn_score) # "Softmax"
        attn_weight = F.softmax(attn_score, dim=2) # "Softmax"
        attn_weight = self.dropout(attn_weight) # Not in the paper

        x = torch.einsum("blmn,bmdn->bldn", attn_weight, k) # "MatMul"
        x = rearrange(x, pattern="b l d n -> b l (d n)")

        x = self.out_proj(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, activ: Literal["relu", "gelu"]="relu", drop_prob=DROP_PROB):
        super().__init__()

        assert activ in ["relu", "gelu"],\
            """`activ` must be one of (`"relu"`, `"gelu"`)"""

        self.dim = dim
        self.mlp_dim = mlp_dim
        self.activ = activ

        self.proj1 = nn.Linear(dim, self.mlp_dim) # $W_{1}$
        if activ == "relu":
            self.relu = nn.ReLU()
        else:
            self.gelu = nn.GELU()
        self.proj2 = nn.Linear(self.mlp_dim, dim) # $W_{2}$
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj1(x)
        if self.activ == "relu":
            x = self.relu(x)
        else:
            x = self.gelu(x)
        x = self.proj2(x)
        x = self.dropout(x) # Not in the paper
        return x


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, dim, mlp_dim, activ, drop_prob=DROP_PROB):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.self_attn = MultiHeadAttention(dim=dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(dim)

        self.ff = PositionwiseFeedForward(dim=dim, mlp_dim=mlp_dim, activ=activ)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(q=x, k=x, v=x, mask=mask) # "Multi-Head Attention" in "Figure 1" of the paper
        x += attn_output # "Add"
        x = self.norm1(x) # "& Norm"

        ff_output = self.ff(x) # "Feed Forward"
        x += ff_output # "Add"
        x = self.norm2(x) # "& Norm"
        x = self.dropout(x) # Not in the paper
        return x


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, src_seq_len, src_pad_idx, n_heads, dim, mlp_dim, n_layers):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.src_seq_len = src_seq_len
        self.src_pad_idx = src_pad_idx
        self.n_heads = n_heads
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.n_layers = n_layers

        self.input = Input(vocab_size=src_vocab_size, dim=dim, pad_idx=src_pad_idx)
        self.enc_stack = nn.ModuleList(
            [EncoderLayer(n_heads=n_heads, dim=dim, mlp_dim=mlp_dim, activ="relu") for _ in range(self.n_layers)]
        )

    def forward(self, x, self_attn_mask):
        x = self.input(x)
        for enc_layer in self.enc_stack:
            x = enc_layer(x, mask=self_attn_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, dim, mlp_dim, activ, drop_prob=DROP_PROB):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim

        self.self_attn = MultiHeadAttention(dim=dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(dim)

        self.enc_dec_attn = MultiHeadAttention(dim=dim, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = PositionwiseFeedForward(dim=dim, mlp_dim=mlp_dim, activ=activ)
        self.norm3 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, enc_output, self_attn_mask, enc_dec_mask):
        attn_output = self.self_attn(q=x, k=x, v=x, mask=self_attn_mask) # "Masked Multi-Head Attention"
            # in "Figure 1" of the paper
        x += attn_output # "Add"
        x = self.norm1(x) # "& Norm"

        attn_output = self.enc_dec_attn(q=x, k=enc_output, v=enc_output, mask=enc_dec_mask) # "Multi-Head Attention"
        x += attn_output # "Add"
        x = self.norm2(x) # "& Norm"

        ff_output = self.ff(x) # "Feed Forward"
        x += ff_output # "Add"
        x = self.norm3(x) # "& Norm"

        x = self.dropout(x) # Not in the paper
        return x


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, trg_seq_len, trg_pad_idx, n_heads, dim, mlp_dim, n_layers):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size
        self.trg_seq_len = trg_seq_len
        self.trg_pad_idx = trg_pad_idx
        self.n_heads = n_heads
        self.dim = dim
        self.n_layers = n_layers

        self.input = Input(vocab_size=trg_vocab_size, dim=dim, pad_idx=trg_pad_idx)
        self.dec_stack = nn.ModuleList(
            [DecoderLayer(n_heads=n_heads, dim=dim, mlp_dim=mlp_dim, activ="relu") for _ in range(self.n_layers)]
        )
        self.linear = nn.Linear(dim, trg_vocab_size)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, enc_output, self_attn_mask=None, enc_dec_mask=None):
        x = self.input(x)
        for dec_layer in self.dec_stack:
            x = dec_layer(x, enc_output=enc_output, self_attn_mask=self_attn_mask, enc_dec_mask=enc_dec_mask)
        x = self.linear(x)
        # x = self.softmax(x)
        x = F.softmax(x, dim=-1)
        return x


def _get_pad_mask(seq, pad_idx=0):
    mask = (seq == pad_idx).unsqueeze(2).unsqueeze(3)
    return mask


# "Prevent positions from attending to subsequent positions." in section 3.1 of the paper
def _get_subsequent_info_mask(src_seq_len, trg_seq_len):
    mask = torch.tril(torch.ones(size=(trg_seq_len, src_seq_len)), diagonal=0).bool()
    mask = mask.unsqueeze(0).unsqueeze(3)
    return mask


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_seq_len,
        trg_seq_len,
        src_pad_idx,
        trg_pad_idx,
        n_heads=N_HEADS,
        dim=D_MODEL,
        mlp_dim=D_MODEL * 4,
        n_layers=N_LAYERS
    ):
        super().__init__()

        assert src_vocab_size == trg_vocab_size, "`src_vocab_size` and `trg_vocab_size` should be equal."
        assert src_seq_len == trg_seq_len, "`src_seq_len` and `trg_seq_len` should be equal."

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_seq_len = src_seq_len
        self.trg_seq_len = trg_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.enc = Encoder(
            src_vocab_size=src_vocab_size,
            src_seq_len=src_seq_len,
            src_pad_idx=src_pad_idx,
            n_heads=n_heads,
            dim=dim,
            mlp_dim=mlp_dim,
            n_layers=n_layers
        )
        self.dec = Decoder(
            trg_vocab_size=trg_vocab_size,
            trg_seq_len=trg_seq_len,
            trg_pad_idx=trg_pad_idx,
            n_heads=n_heads,
            dim=dim,
            mlp_dim=mlp_dim,
            n_layers=n_layers
        )

        # "We share the same weight matrix between the two embedding layers
        # and the pre-softmax linear transformation" in section 3.4 of the paper.
        self.dec.input.embed.weight = self.enc.input.embed.weight
        self.dec.linear.weight = self.dec.input.embed.weight


    def forward(self, src_seq, trg_seq):
        src_pad_mask = _get_pad_mask(seq=src_seq, pad_idx=self.src_pad_idx)
        trg_pad_mask = _get_pad_mask(seq=trg_seq, pad_idx=self.trg_pad_idx)
        trg_subseq_mask = _get_subsequent_info_mask(src_seq_len=self.src_seq_len, trg_seq_len=self.trg_seq_len)

        enc_output = self.enc(src_seq, self_attn_mask=src_pad_mask)
        dec_output = self.dec(
            trg_seq,
            enc_output=enc_output,
            self_attn_mask=trg_pad_mask,
            enc_dec_mask=(trg_pad_mask | trg_subseq_mask) # `&` or `|`??
        )
        return dec_output


if __name__ == "__main__":
    BATCH_SIZE = 16
    SEQ_LEN = 30
    VOCAB_SIZE = 1000
    src_pad_idx = 0
    trg_pad_idx = 0
    transformer = Transformer(
        src_vocab_size=VOCAB_SIZE,
        trg_vocab_size=VOCAB_SIZE,
        src_seq_len=SEQ_LEN,
        trg_seq_len=SEQ_LEN,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx
    )

    src_seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    trg_seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    output = transformer(src_seq=src_seq, trg_seq=trg_seq)
    print(output.shape)
