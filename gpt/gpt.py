# Reference: https://paul-hyun.github.io/gpt-01/?fbclid=IwAR3jaAPdcWBIkShNDr-NIXE5JCfw-UvoQ2h000r5qnSBj8kjrY4ax1jDeM8

import torch
import torch.nn as nn
import torch.functional as F




class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.temperature

        # if mask is not None:
        #     attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        x = torch.matmul(attn_weights, V)
        return x, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=12, d_model=768, dropout=0.1):
        super().__init__()
        # n_heads = 8 # `h`
        d_k = d_v = d_model // n_heads
        
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.scaled_dot_product_attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, Q, K, V, mask=None):
        heads = [
            self.scaled_dot_product_attention(Q=self.W_Q(Q), K=self.W_K(K), V=self.W_V(V))
            for _ in range(self.n_heads)
        ]
        x = torch.cat(heads, axis=2)
        x = self.W_O(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=768, inner_dim=3072):
        super().__init__()

        self.w1 = nn.Linear(d_model, inner_dim)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(inner_dim, d_model)

    def forward(self, x):
        x = self.w1(x)
        x = self.relu(x)
        x = self.w2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_heads=12, d_model=768, n_layers=12):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.multi_head_attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.positional_feed_forward = PositionwiseFeedForward(d_model=d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x, self_attn_mask):
        all_attn_weights = list()
        for _ in range(self.n_layers):
            temp_x, attn_weights = self.multi_head_attn(Q=x, K=x, V=x)
            x += temp_x
            x = self.ln1(x)
            x += self.positional_feed_forward(x)
            x = self.ln2(x)

            all_attn_weights.append(attn_weights)
        return x, all_attn_weights


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


criterion_lm = torch.nn.CrossEntropyLoss()