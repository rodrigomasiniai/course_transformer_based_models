# References
    # https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model

import torch
import torch.nn as nn
import random

from transformer.model import PositionalEncoding, EncoderLayer, get_pad_mask


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, hidden_dim, pad_idx=0):
        super().__init__(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=pad_idx)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, hidden_dim, pad_idx=0):
        super().__init__(3, hidden_dim, padding_idx=pad_idx)


class PositionEmbedding(PositionalEncoding):
    def __init__(self, hidden_dim):
        super().__init__(dim=hidden_dim)


class Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_idx=0, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, hidden_dim=hidden_dim, pad_idx=pad_idx)
        self.seg_embed = SegmentEmbedding(hidden_dim=hidden_dim, pad_idx=pad_idx)
        self.pos_embed = PositionEmbedding(hidden_dim=hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, seg_label):
        x = self.token_embed(seq)
        x = self.pos_embed(x)
        x += self.seg_embed(seg_label)
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.enc_stack = nn.ModuleList(
            [EncoderLayer(d_model=hidden_dim, n_heads=n_heads, activ="gelu") for _ in range(n_layers)]
        )

    def forward(self, x, mask):
        for enc_layer in self.enc_stack:
            x = enc_layer(x, mask=mask)
        return x


class BERT(nn.Module):
    # BERT-BASE: `n_layer=12, hidden_dim=768, n_heads=12`, 110M parameters
    # BERT-LARGE: `n_layer=24, hidden_dim=1024, n_heads=16`, 340M parameters
    def __init__(
        self,
        vocab_size,
        n_layers=12,
        hidden_dim=768,
        n_heads=12,
        n_classes=1000,
        pad_idx=0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.pad_idx = pad_idx

        self.embed = Embedding(vocab_size=vocab_size, hidden_dim=hidden_dim, pad_idx=pad_idx)
        self.tf_enc = TransformerEncoder(n_layers=n_layers, hidden_dim=hidden_dim, n_heads=n_heads)
        self.cls_proj = nn.Linear(HIDDEN_DIM, n_classes)

    def forward(self, seq, seg_label):
        x = self.embed(seq=seq, seg_label=seg_label)

        pad_mask = get_pad_mask(seq=seq, pad_idx=self.pad_idx)
        x = self.tf_enc(x, mask=pad_mask)
        x = self.cls_proj(x[:, 0, :])
        return x


if __name__ == "__main__":
    HIDDEN_DIM = 768
    VOCAB_SIZE = 30_522

    BATCH_SIZE = 16
    SEQ_LEN = 30

    seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    sent1_len = random.randint(0, SEQ_LEN - 1)
    seg_label = torch.as_tensor([0] + [1] * (sent1_len - 1) + [0] + [2] * (SEQ_LEN - sent1_len - 1))

    bert = BERT(vocab_size=VOCAB_SIZE)
    output = bert(seq=seq, seg_label=seg_label)
    print(output.shape)
