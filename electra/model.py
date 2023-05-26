# References
    # https://huggingface.co/docs/transformers/glossary#token-type-ids
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/electra/modeling_electra.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# from bert.model import BERT
from bert.model import (
    TokenEmbedding,
    SegmentEmbedding,
    PositionEmbedding,
    TransformerBlock,
    get_pad_mask,
    MaskedLanguageModelHead,
    # NextSentencePredictionHead
)

DROP_PROB = 0.1
VOCAB_SIZE = 300


class ReplacedTokenDetectionHead(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.cls_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.cls_proj(x)
        return x


class ELECTRAModel(nn.Module):
    def __init__(self, vocab_size, n_layers, hidden_dim, mlp_dim, embed_dim, n_heads, pad_idx=0, drop_prob=DROP_PROB):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.pad_idx = pad_idx
        self.dropout_p = drop_prob

        self.token_embed = TokenEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_idx)
        self.seg_embed = SegmentEmbedding(embed_dim=embed_dim, pad_idx=pad_idx)
        self.pos_embed = PositionEmbedding(embed_dim=embed_dim)
        if hidden_dim != embed_dim:
            self.embed_proj = nn.Linear(embed_dim, hidden_dim)

        self.dropout = nn.Dropout(drop_prob)

        self.tf_block = TransformerBlock(n_layers=n_layers, n_heads=n_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim)

        self.mlm_head = MaskedLanguageModelHead(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.rtd_head = ReplacedTokenDetectionHead(hidden_dim=hidden_dim)

    def forward(self, seq, seg_ids):
        x = self.token_embed(seq)
        x = self.pos_embed(x)
        x += self.seg_embed(seg_ids)
        x = self.dropout(x)
        if self.hidden_dim != self.embed_dim:
            x = self.embed_proj(x)

        pad_mask = get_pad_mask(seq=seq, pad_idx=self.pad_idx)
        x = self.tf_block(x, self_attn_mask=pad_mask)
        return x


class ELECTRA(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers=12,
        disc_hidden_dim=256,
        gen_hidden_dim=64,
        disc_n_heads=4,
        gen_n_heads=1,
        disc_mlp_dim=1024,
        gen_mlp_dim=256,
        embed_dim=128,
        drop_prob=DROP_PROB,
        pad_idx=0
    ):
        super().__init__()

        # vocab_size = VOCAB_SIZE
        # n_layers=12
        # disc_hidden_dim=256
        # gen_hidden_dim=64
        # disc_n_heads=4
        # gen_n_heads=1
        # disc_mlp_dim=1024
        # gen_mlp_dim=256
        # embed_dim=128
        # drop_prob=DROP_PROB
        # pad_idx=0
        disc = ELECTRAModel(
            vocab_size=vocab_size,
            n_layers=n_layers,
            hidden_dim=disc_hidden_dim,
            mlp_dim=disc_mlp_dim,
            embed_dim=embed_dim,
            n_heads=disc_n_heads,
            drop_prob=drop_prob,
            pad_idx=pad_idx
        )
        gen = ELECTRAModel(
            vocab_size=vocab_size,
            n_layers=n_layers,
            hidden_dim=gen_hidden_dim,
            mlp_dim=gen_mlp_dim,
            embed_dim=embed_dim,
            n_heads=gen_n_heads,
            drop_prob=drop_prob,
            pad_idx=pad_idx
        )
        self.tie_weights(gen=self.gen, disc=self.disc)


    def tie_weights(self, gen, disc):
        """
        Weight sharing
        """
        gen.token_embed = disc.token_embed
        gen.pos_embed = disc.pos_embed


    def forward():
        BATCH_SIZE = 8
        SEQ_LEN = 512
        seq = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
        sent1_len = random.randint(0, SEQ_LEN - 1)
        seg_ids = torch.as_tensor([0] + [1] * (sent1_len - 1) + [0] + [2] * (SEQ_LEN - sent1_len - 1))
        logits = gen(seq, seg_ids=seg_ids)

        head = MaskedLanguageModelHead(vocab_size=VOCAB_SIZE, hidden_dim=gen_hidden_dim)
        head(logits).shape


# ELECTRA-Small:
    # `n_layers=12,
    # disc_hidden_dim=256,
    # gen_hidden_dim=64,
    # disc_n_heads=4,
    # gen_n_heads=1,
    # disc_mlp_dim=1024,
    # gen_mlp_dim=256,
    # embed_dim=128,
    # mask_prob=0.15`
# ELECTRA-Base:
    # `n_layers=12,
    # disc_hidden_dim=768,
    # gen_hidden_dim=256,
    # disc_n_heads=12,
    # gen_n_heads=4,
    # disc_mlp_dim=3072,
    # gen_mlp_dim=1024,
    # embed_dim=768,
    # mask_prob=0.15`
# ELECTRA-Large:
    # `n_layers=24,
    # disc_hidden_dim=1024,
    # gen_hidden_dim=256,
    # disc_n_heads=16,
    # gen_n_heads=4,
    # disc_mlp_dim=4096,
    # gen_mlp_dim=1024,
    # embed_dim=1024,
    # mask_prob=0.25`

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# tokenizer(
#     [
#         # ["I love NLP!", "I don't like NLP..."],
#         "I love NLP!",
#         # ["There is an apple.", "I want to eat it."]
#     ],
#     max_length=30,
#     padding="max_length",
#     truncation=True,
#     return_tensors="pt"
# )
# # [101,  1045,  2293, 17953,  2361,   999,   102,  1045,  2123,  1005, 1056,  2066, 17953,  2361,  1012,  1012,  1012,  102]