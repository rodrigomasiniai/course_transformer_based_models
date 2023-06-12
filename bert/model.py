# References
    # https://github.com/codertimo/BERT-pytorch/tree/master/bert_pytorch/model

import torch
import torch.nn as nn

from transformer.model import PositionalEncoding, EncoderLayer, _get_pad_mask

DROP_PROB = 0.1


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim, pad_id=0):
        super().__init__(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=pad_id)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_dim, pad_id=0):
        super().__init__(num_embeddings=2, embedding_dim=embed_dim, padding_idx=pad_id)


class PositionEmbedding(PositionalEncoding):
    def __init__(self, embed_dim):
        super().__init__(dim=embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_dim, mlp_dim, attn_drop_prob, resid_drop_prob):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

        self.enc_stack = nn.ModuleList([
            EncoderLayer(
                n_heads=n_heads,
                dim=hidden_dim,
                mlp_dim=mlp_dim,
                activ="gelu",
                attn_drop_prob=attn_drop_prob,
                resid_drop_prob=resid_drop_prob,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, self_attn_mask):
        for enc_layer in self.enc_stack:
            x = enc_layer(x, mask=self_attn_mask)
        return x


class BERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers=12,
        n_heads=12,
        hidden_dim=768,
        mlp_dim=768 * 4,
        pad_id=0,
        embed_drop_prob=DROP_PROB,
        attn_drop_prob=DROP_PROB,
        resid_drop_prob=DROP_PROB
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.pad_id = pad_id
        self.embed_drop_prob = embed_drop_prob
        self.attn_drop_prob = attn_drop_prob
        self.resid_drop_prob = resid_drop_prob

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, embed_dim=hidden_dim, pad_id=pad_id)
        self.seg_embed = SegmentEmbedding(embed_dim=hidden_dim, pad_id=pad_id)
        self.pos_embed = PositionEmbedding(embed_dim=hidden_dim)

        self.enmbed_drop = nn.Dropout(embed_drop_prob)

        self.tf_block = TransformerBlock(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            attn_drop_prob=attn_drop_prob,
            resid_drop_prob=resid_drop_prob,
        )

    def forward(self, seq, seg_ids):
        x = self.token_embed(seq)
        x = self.pos_embed(x)
        x += self.seg_embed(seg_ids)
        x = self.enmbed_drop(x)

        pad_mask = _get_pad_mask(seq=seq, pad_id=self.pad_id)
        x = self.tf_block(x, self_attn_mask=pad_mask)
        return x


# 110M parameters
class BERTBase(BERT):
    def __init__(self, vocab_size, pad_id=0):
        super().__init__(vocab_size=vocab_size, pad_id=pad_id)


# 340M parameters
class BERTLarge(BERT):
    def __init__(self, vocab_size, pad_id=0):
        super().__init__(
            vocab_size=vocab_size,
            n_layers=24,
            n_heads=16,
            hidden_dim=1024,
            pad_id=pad_id
        )


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
    def __init__(self, vocab_size, hidden_dim=768, drop_prob=DROP_PROB):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.cls_proj = nn.Linear(hidden_dim, vocab_size)
        self.head_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.cls_proj(x)
        x = self.head_drop(x)
        return x


class NextSentencePredictionHead(nn.Module):
    def __init__(self, hidden_dim=768, drop_prob=DROP_PROB):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cls_proj = nn.Linear(hidden_dim, 2)
        self.head_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.cls_proj(x)
        x = self.head_drop(x)
        return x


if __name__ == "__main__":
    HIDDEN_DIM = 768
    VOCAB_SIZE = 30_522

    BATCH_SIZE = 8
    SEQ_LEN = 512
    seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    sent1_len = torch.randint(low=2, high=SEQ_LEN + 1, size=(BATCH_SIZE,))
    seg_ids = torch.as_tensor([[0] * i + [1] * (SEQ_LEN - i) for i in sent1_len], dtype=torch.int64)

    model = BERTBase(vocab_size=VOCAB_SIZE)
    # model = BERTLarge(vocab_size=VOCAB_SIZE)
    output = model(seq=seq, seg_ids=seg_ids)
    print(output.shape)
