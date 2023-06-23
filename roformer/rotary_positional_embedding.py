# References
    # https://nn.labml.ai/transformers/rope/index.html

import sys
import torch
from torch import nn

# "Rotary encoding transforms pairs of features by rotating in the 2D plane. That is, it organizes the $d$ features
# as $d/2$ pairs. Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
# by an angle depending on the position of the token."

# "We pair feature $i$ with feature $i + 2/d$. So for position $m$ we transform
# $$\left(\begin{matrix}
#     x^{(i)}_{m}\\
#     x^{(i + d / 2)}_{m}\\
# \end{matrix}\right)$$
# to
# $$\left(\begin{matrix}
#     x^{(i)}_{m}\cos{m\theta_{i}} - x^{(i + d / 2)}_{m}\sin{m\theta_{i}}\\
#     x^{(i + d / 2)}_{m}\cos{m\theta_{i}} + x^{(i)}_{m}\sin{m\theta_{i}}\\
# \end{matrix}\right)$$"

# $⟨RoPE(xm(1)​,xm(2)​,m),RoPE(xn(1)​,xn(2)​,n)⟩$
# This shows that for dot-production attention the rotary encodings gives relative attention.

torch.set_printoptions(edgeitems=16, linewidth=sys.maxsize, sci_mode=True)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()

        self.dim = dim
        self.base = base

    def _get_theta(self, i):
        # $\Theta = \{\theta_{i} = 10000^{-2(i - 1)/d}, i \in [1, 2, \ldots d/2]\}$
        return self.base ** (-2 * (i - 1) / self.dim)

    # `x`` is the tensor at the head of a key or a query with shape (`batch_size`, `n_heads`, `seq_len`, `dim`)
    def forward(self, x):
        # x = torch.randn((16, 512, 64))
        b, n, seq_len, d = x.shape

        pos = torch.arange(seq_len, dtype=x.dtype)
        i = torch.arange(1, self.dim // 2 + 1).repeat(2)
        theta = self._get_theta(i) # $\theta_{i}$
        v = torch.einsum("p,t->pt", pos, theta)

        self.sin_cached = torch.sin(v)
        self.cos_cached = torch.cos(v)

rope = RotaryPositionalEmbedding(d=64)
x = torch.randn((512, 64))
rope(x)
rope.cos_cached


# class RotaryPositionalEmbeddings(nn.Module):
#     def __init__(self, d: int, base: int = 10_000):
#         """
#         * `d` is the number of features $d$
#         * `base` is the constant used for calculating $\Theta$
#         """
#         super().__init__()

#         base = base
#         d = d
#         cos_cached = None
#         sin_cached = None

#     def _build_cache(self, x: torch.Tensor):
#         """
#         Cache $\cos$ and $\sin$ values
#         """
#         # Return if cache is already built
#         if cos_cached is not None and x.shape[0] <= cos_cached.shape[0]:
#             return

#         # Get sequence length
#         seq_len = x.shape[0]

# $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
theta = 1. / (base ** (torch.arange(0, d, 2).float() / d)).to(x.device)
seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
idx_theta2
idx_theta.shape

cos_cached = idx_theta2.cos()[:, None, None, :]
sin_cached = idx_theta2.sin()[:, None, None, :]

#     def _neg_half(self, x: torch.Tensor):
#         # $\frac{d}{2}$
#         d_2 = d // 2

#         # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
#         return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

#     def forward(self, x: torch.Tensor):
#         """
#         * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
#         """
#         # Cache $\cos$ and $\sin$ values
#         _build_cache(x)

#         # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
#         x_rope, x_pass = x[..., :d], x[..., d:]

#         # Calculate
#         # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
#         neg_half_x = _neg_half(x_rope)

#         # Calculate
#         #
#         # \begin{align}
#         # \begin{pmatrix}
#         # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
#         # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
#         # \end{pmatrix} \\
#         # \end{align}
#         #
#         # for $i \in {1, 2, ..., \frac{d}{2}}$
#         x_rope = (x_rope * cos_cached[:x.shape[0]]) + (neg_half_x * sin_cached[:x.shape[0]])

#         #
#         return torch.cat((x_rope, x_pass), dim=-1)