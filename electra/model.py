import torch
import torch.nn as nn
import torch.nn.functional as F

from bert.model import BERT

DROPOUT_PROB = 0.1


class ELECTRA(nn.Module):
    # ELECTRA-Small: `n_layers=12, hidden_dim=256, n_heads=4, mlp_dim=1024`
    # ELECTRA-Base: `n_layers=12, hidden_dim=768, n_heads=12, mlp_dim=3072`
    # ELECTRA-Large: `n_layers=24, hidden_dim=1024, n_heads=16, mlp_dim=4096`
    def __init__(
        self,
        vocab_size,
        n_layers=12,
        hidden_dim=768,
        n_heads=12,
        pad_idx=0,
        dropout_prob=DROPOUT_PROB
    ):
        super().__init__()


DISC_HIDDEN_DIM, GEN_HIDDEN_DIM = 256, 64 # 1/4 Small
DISC_HIDDEN_DIM, GEN_HIDDEN_DIM = 768, 256 # 1/3 Base
DISC_HIDDEN_DIM, GEN_HIDDEN_DIM = 1024, 256 # 1/4 Large

DISC_MLP_DIM, GEN_MLP_DIM = 1024, 256
3072, 1024
4096, 1024

DISC_N_HEADS, GEN_N_HEADS = 4, 1
12, 4
16, 4

