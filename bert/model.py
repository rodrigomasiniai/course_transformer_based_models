# References
    # https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model

import torch
import torch.nn as nn
import random

from transformer.model import PositionalEncoding, EncoderLayer, get_pad_mask

DROP_PROB = 0.1


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim, pad_idx=0):
        super().__init__(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_idx)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_dim, pad_idx=0):
        # `num_embeddings=3`: `"[PAD]"` 토큰 때문!
        super().__init__(num_embeddings=3, embedding_dim=embed_dim, padding_idx=pad_idx)


class PositionEmbedding(PositionalEncoding):
    def __init__(self, embed_dim):
        super().__init__(dim=embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_dim, mlp_dim):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        self.enc_stack = nn.ModuleList(
            [
                EncoderLayer(n_heads=n_heads, dim=hidden_dim, mlp_dim=mlp_dim, activ="gelu", drop_prob=DROP_PROB)
                for _
                in range(n_layers)
            ]
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
        n_heads=12,
        hidden_dim=768,
        mlp_dim=768 * 4,
        pad_idx=0,
        DROP_PROB=DROP_PROB
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.pad_idx = pad_idx

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, embed_dim=hidden_dim, pad_idx=pad_idx)
        self.seg_embed = SegmentEmbedding(embed_dim=hidden_dim, pad_idx=pad_idx)
        self.pos_embed = PositionEmbedding(embed_dim=hidden_dim)

        self.dropout = nn.Dropout(DROP_PROB)

        self.tf_block = TransformerBlock(n_layers=n_layers, n_heads=n_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim)

    def forward(self, seq, seg_ids):
        x = self.token_embed(seq)
        x = self.pos_embed(x)
        x += self.seg_embed(seg_ids)
        x = self.dropout(x)

        pad_mask = get_pad_mask(seq=seq, pad_idx=self.pad_idx)
        x = self.tf_block(x, self_attn_mask=pad_mask)
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
    seg_ids = torch.as_tensor([0] + [1] * (sent1_len - 1) + [0] + [2] * (SEQ_LEN - sent1_len - 1))

    bert = BERT(vocab_size=VOCAB_SIZE)
    output = bert(seq=seq, seg_ids=seg_ids)
    print(output.shape)
