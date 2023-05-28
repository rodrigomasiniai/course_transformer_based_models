import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


EMBED_DIM = 768
# embed1 = torch.randn(size=(BATCH_SIZE, EMBED_DIM))
# embed2 = torch.randn(size=(BATCH_SIZE, EMBED_DIM))

class SentenceBERT(nn.Module):
    def __init__(
        self,
        bert,
        n_labels=3,
        embed_dim=EMBED_DIM,
        training=False,
        pooling: Literal["mean", "max", "cls"]="mean",
        objective: Literal["classification", "regression", "triplet"]="classification",
    ):
        super().__init__()

        self.bert = bert
        self.n_labels = n_labels
        self.training = training
        self.pooling = pooling
        self.objective = objective
    
        if objective == "classification":
            # "Classification Object Function" in section 3 of the paper
            self.proj = nn.Linear(embed_dim * 3, n_labels) # $W_{t}$
            self.softmax = nn.Softmax(dim=1)

    def forward(self, sent1, sent2):
        x1, x2 = bert(sent1, seg_ids=torch.zeros_like(sent1)), bert(sent2, seg_ids=torch.zeros_like(sent2))

        if self.pooling == "mean":
            x1, x2 = x1[:, 1: -1, :], x2[:, 1: -1, :]
            x1, x2 = torch.max(x1, dim=1)[0], torch.max(x2, dim=1)[0]
        elif self.pooling == "mean":
            x1, x2 = x1[:, 1: -1, :], x2[:, 1: -1, :]
            x1, x2 = torch.mean(x1, dim=1), torch.mean(x2, dim=1)
        else:
            x1, x2 = x1[:, 0, :], x2[:, 0, :]

        if self.objective == "classification":
            x = torch.cat([x1, x2, torch.abs(x1 - x2)], dim=1)
            x = self.proj(x)
            x = self.softmax(x)
        # elif objective == "regression":
        return x
VOCAB_SIZE = 30_522
bert = BERT(vocab_size=VOCAB_SIZE)
model = SentenceBERT(bert=bert)

BATCH_SIZE = 4
SEQ_LEN = 512
sent1 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
sent2 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
model(sent1=sent1, sent2=sent2)

model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')