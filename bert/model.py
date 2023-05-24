# References
    # https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model

import torch
import torch.nn as nn
import random

from transformer.model import PositionalEncoding, EncoderLayer, get_pad_mask

DROPOUT_PROB = 0.1


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, hidden_dim, pad_idx=0):
        super().__init__(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=pad_idx)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, hidden_dim, pad_idx=0):
        # `num_embeddings=3`: `"[PAD]"` 토큰 때문!
        super().__init__(num_embeddings=3, embedding_dim=hidden_dim, padding_idx=pad_idx)


class PositionEmbedding(PositionalEncoding):
    def __init__(self, hidden_dim):
        super().__init__(dim=hidden_dim)


# class Embedding(nn.Module):
#     def __init__(self, vocab_size, hidden_dim, pad_idx=0, dropout_prob=DROPOUT_PROB):
#         super().__init__()

#         self.vocab_size = vocab_size
#         self.hidden_dim = hidden_dim

#         self.token_embed = TokenEmbedding(vocab_size=vocab_size, hidden_dim=hidden_dim, pad_idx=pad_idx)
#         self.seg_embed = SegmentEmbedding(hidden_dim=hidden_dim, pad_idx=pad_idx)
#         self.pos_embed = PositionEmbedding(hidden_dim=hidden_dim)

#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, seq, seg_label):
#         x = self.token_embed(seq)
#         x = self.pos_embed(x)
#         x += self.seg_embed(seg_label)
#         x = self.dropout(x)
#         return x


class TransformerBlock(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.enc_stack = nn.ModuleList(
            [EncoderLayer(d_model=hidden_dim, n_heads=n_heads, activ="gelu") for _ in range(n_layers)]
        )

    def forward(self, x, self_attn_mask):
        for enc_layer in self.enc_stack:
            x = enc_layer(x, mask=self_attn_mask)
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
        pad_idx=0,
        dropout_prob=DROPOUT_PROB
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.pad_idx = pad_idx

        # self.embed = Embedding(vocab_size=vocab_size, hidden_dim=hidden_dim, pad_idx=pad_idx)
        self.token_embed = TokenEmbedding(vocab_size=vocab_size, hidden_dim=hidden_dim, pad_idx=pad_idx)
        self.seg_embed = SegmentEmbedding(hidden_dim=hidden_dim, pad_idx=pad_idx)
        self.pos_embed = PositionEmbedding(hidden_dim=hidden_dim)

        self.dropout = nn.Dropout(dropout_prob)

        self.tf_enc = TransformerBlock(n_layers=n_layers, hidden_dim=hidden_dim, n_heads=n_heads)

    def forward(self, seq, seg_label):
        # x = self.embed(seq=seq, seg_label=seg_label)
        x = self.token_embed(seq)
        x = self.pos_embed(x)
        x += self.seg_embed(seg_label)
        x = self.dropout(x)

        pad_mask = get_pad_mask(seq=seq, pad_idx=self.pad_idx)
        x = self.tf_enc(x, self_attn_mask=pad_mask)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim=768, n_classes=1000):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_classed = n_classes

        self.cls_proj = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.cls_proj(x)
        return x


class MaskedLanguageModelHead(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.cls_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.cls_proj(x)
        return x


class NextSentencePredictionHead(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cls_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.cls_proj(x)
        return x


if __name__ == "__main__":
    HIDDEN_DIM = 768
    VOCAB_SIZE = 30_522

    BATCH_SIZE = 8
    SEQ_LEN = 512

    seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    sent1_len = random.randint(0, SEQ_LEN - 1)
    seg_label = torch.as_tensor([0] + [1] * (sent1_len - 1) + [0] + [2] * (SEQ_LEN - sent1_len - 1))

    bert = BERT(vocab_size=VOCAB_SIZE)
    output = bert(seq=seq, seg_label=seg_label)
    print(output.shape)
