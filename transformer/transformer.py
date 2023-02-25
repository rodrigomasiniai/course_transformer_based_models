import torch
import torch.nn as nn
import torch.nn.functional as F


d_word_vec=512,
d_inner=2048,
n_layers=6,
d_model=512,
d_k=64,
d_v=64,
dropout=0.1,
n_position=200,
trg_emb_prj_weight_sharing=True,
emb_src_trg_weight_sharing=True


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # attn_scores = torch.matmul(q, k.transpose(dim0=2, dim1=3)) / self.temperature
        attn_scores = torch.matmul(q, k.transpose(dim0=0, dim1=1)) / self.temperature

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        dropout = self.dropout(attn_weights)
        output = torch.matmul(dropout, v)
        return output, dropout


d_model=512
h=8
d_k=d_model // h
d_v=d_model // h
scaled_dot_product_attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
q = torch.rand((1, d_model))
k = torch.rand((1, d_model))
v = torch.rand((1, d_model))
scaled_dot_product_attention(q=q, k=k, v=v)

w_qs = nn.Linear(d_model, h * d_k, bias=False)
w_qs(q).shape

W_V = nn.Linear(d_model, d_v, bias=False)
W_Q(q), W_K(k), W_V(v)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, h * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, h * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, h * d_v, bias=False)
        self.fc = nn.Linear(h * d_v, d_model, bias=False)
        
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.W_O = nn.Linear(h * d_v, d_model, bias=False)

        self.head = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    # `Q`: `(batch_size, len_Q, d_model)`
    # `K`: `(batch_size, len_K, d_model)`
    # `V`: `(batch_size, len_V, d_model)`
    def forward(self, Q, K, V, mask=None):

        # d_k, d_v, h = self.d_k, self.d_v, self.h
        batch_size, len_Q, len_K, len_V = Q.size(0), Q.size(1), K.size(1), V.size(1)

        residual = Q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q = self.w_qs(q).view(batch_size, len_Q, h, d_k)
        # k = self.w_ks(k).view(batch_size, len_K, h, d_k)
        # v = self.w_vs(v).view(batch_size, len_V, h, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.head(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_Q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        return q, attn


# class MultiHeadAttention(nn.Module):
#     def __init__(self, h, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()

#         self.h = h
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, h * d_k, bias=False)
#         self.w_ks = nn.Linear(d_model, h * d_k, bias=False)
#         self.w_vs = nn.Linear(d_model, h * d_v, bias=False)
#         self.fc = nn.Linear(h * d_v, d_model, bias=False)

#         self.head = ScaledDotProductAttention(temperature=d_k ** 0.5)

#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

#     def forward(self, q, k, v, mask=None):

#         d_k, d_v, h = self.d_k, self.d_v, self.h
#         batch_size, len_Q, len_K, len_V = q.size(0), q.size(1), k.size(1), v.size(1)

#         residual = q

#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_qs(q).view(batch_size, len_Q, h, d_k)
#         k = self.w_ks(k).view(batch_size, len_K, h, d_k)
#         v = self.w_vs(v).view(batch_size, len_V, h, d_v)

#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

#         if mask is not None:
#             mask = mask.unsqueeze(1)   # For head axis broadcasting.

#         q, attn = self.head(q, k, v, mask=mask)

#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = q.transpose(1, 2).contiguous().view(batch_size, len_Q, -1)
#         q = self.dropout(self.fc(q))
#         q += residual

#         q = self.layer_norm(q)
#         return q, attn