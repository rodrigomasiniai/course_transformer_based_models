<!-- ## (Masked) Multi-head Self Attention
- 논문에 충실하게 구현하면 다음과 같습니다.
```python
class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads # $h$
        self.d_model = d_model

        self.d_kv = d_model // n_heads # $d_{k}$, $d_{v}$

        self.W_Q = nn.Linear(d_model, self.d_kv, bias=False)
        self.W_K = nn.Linear(d_model, self.d_kv, bias=False)
        self.W_V = nn.Linear(d_model, self.d_kv, bias=False)
        self.W_O = nn.Linear(n_heads * self.d_kv, self.d_model, bias=False)

        self.scaled_dot_product_attn = ScaledDotProductAttention(temperature=self.d_kv ** 0.5)

    def forward(self, seq, mask=None):
        heads = list()
        tot_attn_weights = list()
        for _ in range(self.n_heads):
            x, attn_weights = self.scaled_dot_product_attn(Q=self.W_Q(seq), K=self.W_K(seq), V=self.W_V(seq))
            heads.append(x)
            tot_attn_weights.append(attn_weights)
        x = torch.cat(heads, dim=2)
        x = self.W_O(x)
        return x
``` -->