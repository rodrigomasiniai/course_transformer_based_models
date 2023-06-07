import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
from typing import Literal

from bert.model import BERTBase


def _perform_sentence_bert_pooler(x, pooler):
    if pooler == "mean":
        x = x[:, 1: -1, :]
        x = torch.max(x, dim=1)[0]
    elif pooler == "mean":
        x = x[:, 1: -1, :]
        x = torch.mean(x, dim=1)
    else:
        x = x[:, 0, :]
    return x


class SentenceBERTForClassification(nn.Module):
    def __init__(self, embedder, pooler: Literal["mean", "max", "cls"]="mean"):
        super().__init__()

        self.embedder = embedder
        self.pooler = pooler

        self.proj = nn.Linear(embedder.hidden_dim * 3, 3) # $W_{t}$ in the paper.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent1, sent2):
        x1, x2 = (
            self.embedder(sent1, seg_ids=torch.zeros_like(sent1)),
            self.embedder(sent2, seg_ids=torch.zeros_like(sent2))
        )
        x1, x2 = (
            _perform_sentence_bert_pooler(x1, pooler=self.pooler),
            _perform_sentence_bert_pooler(x2, pooler=self.pooler)
        )
        x = torch.cat([x1, x2, torch.abs(x1 - x2)], dim=1)
        x = self.proj(x)
        x = self.softmax(x)
        return x

    def _get_finetuned_embedder(self):
        return self.embedder


class SentenceBERTForRegression(nn.Module):
    def __init__(self, embedder, pooler: Literal["mean", "max", "cls"]="mean"):
        super().__init__()

        self.embedder = embedder
        self.pooler = pooler

    def forward(self, sent1, sent2):
        x1, x2 = (
            self.embedder(sent1, seg_ids=torch.zeros_like(sent1)),
            self.embedder(sent2, seg_ids=torch.zeros_like(sent2))
        )
        x1, x2 = (
            _perform_sentence_bert_pooler(x1, pooler=self.pooler),
            _perform_sentence_bert_pooler(x2, pooler=self.pooler)
        )
        x = F.cosine_similarity(x1, x2)
        return x

    def _get_finetuned_embedder(self):
        return self.embedder


class SentenceBERTWithTripletNetworks(nn.Module):
    def __init__(self, embedder, pooler: Literal["mean", "max", "cls"]="mean", eps=1):
        super().__init__()

        self.embedder = embedder
        self.pooler = pooler
        self.eps = eps

    def forward(self, a, p, n):
        a, p, n = (
            self.embedder(a, seg_ids=torch.zeros_like(a)),
            self.embedder(p, seg_ids=torch.zeros_like(p)),
            self.embedder(n, seg_ids=torch.zeros_like(n))
        )
        a, p, n = (
            _perform_sentence_bert_pooler(a, pooler=self.pooler),
            _perform_sentence_bert_pooler(p, pooler=self.pooler),
            _perform_sentence_bert_pooler(n, pooler=self.pooler)
        )
        return a, p, n
        # x = LA.vector_norm(a - p, ord=2, dim=1) - LA.vector_norm(a - n, ord=2, dim=1) + self.eps
        # x = F.relu(x)
        # return x

    def _get_finetuned_embedder(self):
        return self.embedder


if __name__ == "__main__":
    VOCAB_SIZE = 30_522
    bert = BERTBase(vocab_size=VOCAB_SIZE)

    BATCH_SIZE = 4
    SEQ_LEN = 128
    cls_sbert = SentenceBERTForClassification(embedder=bert)
    sent1 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    sent2 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    output = cls_sbert(sent1=sent1, sent2=sent2)
    print(output)

    reg_sbert = SentenceBERTForRegression(embedder=bert)
    output = reg_sbert(sent1=sent1, sent2=sent2)
    print(output)

    trip_sbert = SentenceBERTWithTripletNetworks(embedder=bert)
    a = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    p = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    n = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    output = trip_sbert(a=a, p=p, n=n)
    print(output)
