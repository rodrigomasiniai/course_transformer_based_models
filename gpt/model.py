# References:
    # https://paul-hyun.github.io/gpt-01/?fbclid=IwAR3jaAPdcWBIkShNDr-NIXE5JCfw-UvoQ2h000r5qnSBj8kjrY4ax1jDeM8
    # https://gaussian37.github.io/dl-pytorch-lr_scheduler/

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler




class GPT(nn.Module):
    def __init__(self, vocab_size, max_len=512, n_heads=12, d_model=768, n_layers=12):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.position_embedding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.decoder = Decoder(n_heads=n_heads, d_model=d_model, n_layers=n_layers)
    
    def forward(self, x):
        x = self.embedding(x)
        x += self.position_embedding(x)
        x, attn_weights = self.decoder(x)
        return x, attn_weights
