# References:
    # https://paul-hyun.github.io/gpt-01/?fbclid=IwAR3jaAPdcWBIkShNDr-NIXE5JCfw-UvoQ2h000r5qnSBj8kjrY4ax1jDeM8
    # https://gaussian37.github.io/dl-pytorch-lr_scheduler/

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


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



class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))

        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                for base_lr
                in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


# We used the Adam optimization scheme with a max learning rate of 2.5e-4. The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.
max_lr = 2.5e-4
batch_size = 64
n_epochs = 100
gpt = GPT()
optimizer = optim.Adam(params=gpt.parameters(), lr=0)
scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_up=2_000, eta_max=2.5e-4, T_0=150, T_mult=1, gamma=1)
