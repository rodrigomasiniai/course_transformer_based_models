# Reference:
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
    # https://github.com/huggingface/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/vision_transformer.py#L216
    # https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)
    # https://wikidocs.net/31379

import torch
import torch.nn as nn
from einops import rearrange

N_HEADS = 8
D_MODEL = 512
N_LAYERS = 6
D_FF = 2048


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int=5000) -> None:
        super().__init__()

        pos = torch.arange(max_len).unsqueeze(1)
        i = torch.arange(dim // 2).unsqueeze(0)
        angle = pos / (10_000 ** (2 * i / dim))

        self.pe = torch.zeros(size=(max_len, dim))
        self.pe[:, 0:: 2] = torch.sin(angle)
        self.pe[:, 1:: 2] = torch.cos(angle)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape `[batch_size, seq_len, embedding_dim]`
        """
        b, l, _ = x.shape
        x += self.pe.unsqueeze(0)[:, : l, :]
        return x


class Input(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(dim=d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embed(x)
        # "In the embedding layers, we multiply those weights by $\sqrt{d_{text{model}}}$."
        # in section 3.4 of the paper.
        x *= self.d_model ** 0.5
        x = self.pos_enc(x)
        x = self.dropout(x) # Not in the paper
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=D_MODEL, n_heads=N_HEADS):
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

    def forward(self, q, k, v, mask=None):
        b, l, _ = q.shape
        _, m, _ = k.shape

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(b, l, self.head_dim, self.n_heads)
        k = k.view(b, m, self.head_dim, self.n_heads)
        v = v.view(b, m, self.head_dim, self.n_heads)

        attn_score = torch.einsum("bldn,bmdn->blmn", q, k) # "MatMul" in "Figure 2" of the paper
        if mask is not None:
            attn_score.masked_fill_(mask=mask, value=-1e9) # "Mask (opt.)"
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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.w1(x)
        x = self.relu(x)
        x = self.w2(x)
        x = self.dropout(x) # Not in the paper
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

    def forward(self, x, mask):
        attn_output = self.self_attn(q=x, k=x, v=x, mask=mask) # "Multi-Head Attention" in "Figure 1" of the paper
        x += attn_output # "Add"
        x = self.norm1(x) # "& Norm"

        ff_output = self.ff(x) # "Feed Forward"
        x += ff_output # "Add"
        x = self.norm2(x) # "& Norm"
        x = self.dropout(x) # Not in the paper
        return x


class Encoder(nn.Module):
    def __init__(
        self, src_vocab_size, src_seq_len, src_pad_idx, n_heads=N_HEADS, d_model=D_MODEL, n_layers=N_LAYERS
    ):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.src_seq_len = src_seq_len
        self.src_pad_idx = src_pad_idx
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.input = Input(vocab_size=src_vocab_size, d_model=d_model, pad_idx=src_pad_idx)
        self.enc_stack = nn.ModuleList(
            [EncoderLayer(d_model=d_model, n_heads=n_heads) for _ in range(self.n_layers)]
        )

    def forward(self, x, mask):
        x = self.input(x)
        for enc_layer in self.enc_stack:
            x = enc_layer(x, mask=mask)
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

    def forward(self, x, enc_output, mask):
        attn_output = self.self_attn(q=x, k=x, v=x) # "Masked Multi-Head Attention" in "Figure 1" of the paper
        x += attn_output # "Add"
        x = self.norm1(x) # "& Norm"

        attn_output = self.enc_dec_attn(q=x, k=enc_output, v=enc_output, mask=mask) # "Multi-Head Attention"
        x += attn_output # "Add"
        x = self.norm2(x) # "& Norm"

        ff_output = self.ff(x) # "Feed Forward"
        x += ff_output # "Add"
        x = self.norm3(x) # "& Norm"

        x = self.dropout(x) # Not in the paper
        return x


class Decoder(nn.Module):
    def __init__(
        self, trg_vocab_size, trg_seq_len, trg_pad_idx, n_heads=N_HEADS, d_model=D_MODEL, n_layers=N_LAYERS
    ):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size
        self.trg_seq_len = trg_seq_len
        self.trg_pad_idx = trg_pad_idx
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.input = Input(vocab_size=trg_vocab_size, d_model=d_model, pad_idx=trg_pad_idx)
        self.dec_stack = nn.ModuleList(
            [DecoderLayer(n_heads=n_heads, d_model=d_model) for _ in range(self.n_layers)]
        )
        self.linear = nn.Linear(d_model, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, enc_output, mask):
        x = self.input(x)
        for dec_layer in self.dec_stack:
            x = dec_layer(x, enc_output=enc_output, mask=mask)
        x = self.linear(x)
        x = self.softmax(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_seq_len,
        trg_seq_len,
        src_pad_idx,
        trg_pad_idx
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
            src_pad_idx=src_pad_idx
        )
        self.dec = Decoder(
            trg_vocab_size=trg_vocab_size,
            trg_seq_len=trg_seq_len,
            trg_pad_idx=trg_pad_idx
        )

        # "We share the same weight matrix between the two embedding layers and the pre-softmax linear transformation"
        # in # section 3.4 of the paper.
        self.dec.input.embed.weight = self.enc.input.embed.weight
        self.dec.linear.weight = self.dec.input.embed.weight

    def _get_pad_mask(self, seq, pad_idx):
        mask = (seq == pad_idx).unsqueeze(2).unsqueeze(3)
        return mask

    def _get_subsequent_info_mask(self):
        # "Prevent positions from attending to subsequent positions." in section 3.1 of the paper
        mask = torch.tril(torch.ones(size=(self.trg_seq_len, self.src_seq_len)), diagonal=0).bool()
        mask = mask.unsqueeze(0).unsqueeze(3)
        return mask

    def forward(self, src_seq, trg_seq):
        src_pad_mask = self._get_pad_mask(seq=src_seq, pad_idx=self.src_pad_idx)
        trg_pad_mask = self._get_pad_mask(seq=trg_seq, pad_idx=self.trg_pad_idx)
        trg_subseq_mask = self._get_subsequent_info_mask()
        trg_mask = (trg_pad_mask | trg_subseq_mask)

        enc_output = self.enc(src_seq, mask=src_pad_mask)
        dec_output = self.dec(trg_seq, enc_output=enc_output, mask=trg_mask)
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
