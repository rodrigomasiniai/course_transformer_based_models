import torch
import torch.nn as nn
import torch.nn.functional as F


batch_size = 4
vocab_size = 1_000
seq_len = 380
d_model=512



class ResidualConnection(nn.Module):
    def __init__(self, fn):
        super().__init__()

        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    # def forward(self, q, k, v, mask=None):
    def forward(self, Q, K, V, mask=None):
        # attn_scores = torch.matmul(q, k.transpose(dim0=0, dim1=1)) / self.temperature
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.temperature

        # if mask is not None:
        #     attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # attn_weights = F.softmax(attn_scores, dim=-1)
        # dropout = self.dropout(attn_weights)
        # x = torch.matmul(dropout, v)
        attn_weights = F.softmax(attn_scores, dim=-1)
        x = torch.matmul(attn_weights, K)
        # return x, dropout
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        n_heads = 8 # `h`
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
            scaled_dot_product_attention(Q=self.W_Q(Q), K=self.W_K(K), V=self.W_V(V))
            for _ in range(self.n_heads)
        ]
        x = torch.cat(heads, axis=2)
        x = self.W_O(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.w1(x)
        x = self.relu(x)
        x = self.w2(x)
        return x


class Encoder(nn.Module):
    # `n_layers`: $N$
    def __init__(self, n_heads, d_model, n_layers=6):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers

        self.multi_head_attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model)
        # residual_connection = ResidualConnection(fn=multi_head_attn)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        for _ in range(n_layers):
            # multi_head_attn(Q=x, K=x, V=x).shape
            x += self.multi_head_attn(Q=x, K=x, V=x)
            x = self.norm1(x)
            x += self.feed_forward(x)
            x = self.norm2(x)
        return x


input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
x = embedding(input)
x.shape
# positional_encoding = nn.PositionalEncoding()
# x = positional_encoding(x)



d_word_vec=512
d_inner=2048
n_layers=6
d_k=64
d_v=64
dropout=0.1
n_position=200
trg_emb_prj_weight_sharing=True
emb_src_trg_weight_sharing=True


