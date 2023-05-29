import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


EMBED_DIM = 768
# embed1 = torch.randn(size=(BATCH_SIZE, EMBED_DIM))
# embed2 = torch.randn(size=(BATCH_SIZE, EMBED_DIM))


def _perform_sentence_bert_pooling(x1, x2, pooling):
    if pooling == "mean":
        x1, x2 = x1[:, 1: -1, :], x2[:, 1: -1, :]
        x1, x2 = torch.max(x1, dim=1)[0], torch.max(x2, dim=1)[0]
    elif pooling == "mean":
        x1, x2 = x1[:, 1: -1, :], x2[:, 1: -1, :]
        x1, x2 = torch.mean(x1, dim=1), torch.mean(x2, dim=1)
    else:
        x1, x2 = x1[:, 0, :], x2[:, 0, :]
    return x1, x2


class ClassificationSentenceBERT(nn.Module):
    def __init__(
        self,
        bert,
        embed_dim=EMBED_DIM,
        pooling: Literal["mean", "max", "cls"]="mean",
    ):
        super().__init__()

        self.bert = bert
        self.pooling = pooling
    
        # "cls Object Function" in section 3 of the paper
        self.proj = nn.Linear(embed_dim * 3, 3) # $W_{t}$
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent1, sent2):
        x1, x2 = bert(sent1, seg_ids=torch.zeros_like(sent1)), bert(sent2, seg_ids=torch.zeros_like(sent2))
        x1, x2 = _perform_sentence_bert_pooling(x1, x2, pooling=self.pooling)
        x = torch.cat([x1, x2, torch.abs(x1 - x2)], dim=1)
        x = self.proj(x)
        x = self.softmax(x)
        return x

    def _get_trained_model(self):
        return self.bert


class RegressionSentenceBERT(nn.Module):
    def __init__(
        self,
        bert,
        pooling: Literal["mean", "max", "cls"]="mean",
    ):
        super().__init__()

        self.bert = bert
        self.pooling = pooling

    def forward(self, sent1, sent2):
        x1, x2 = bert(sent1, seg_ids=torch.zeros_like(sent1)), bert(sent2, seg_ids=torch.zeros_like(sent2))
        x1, x2 = _perform_sentence_bert_pooling(x1, x2, pooling=self.pooling)
        x = F.cosine_similarity(x1, x2)
        return x

    def _get_trained_model(self):
        return self.bert


class TripletSentenceBERT(nn.Module):
    def __init__(
        self,
        bert,
        pooling: Literal["mean", "max", "cls"]="mean",
        eps=1
    ):
        super().__init__()

        self.bert = bert
        self.pooling = pooling
        self.eps = eps

    def forward(self, anchor, pos, neg):
        a, p, n = (
            bert(anchor, seg_ids=torch.zeros_like(anchor)),
            bert(pos, seg_ids=torch.zeros_like(pos)),
            bert(neg, seg_ids=torch.zeros_like(neg))
        )
        x = torch.norm(a - p, p=2, dim=1) - torch.norm(a - n, p=2, dim=1) + self.eps
        return x

    def _get_trained_model(self):
        return self.bert


if __name__ == "__main__":
    VOCAB_SIZE = 30_522
    bert = BERT(vocab_size=VOCAB_SIZE)
    cls_sbert = ClassificationSentenceBERT(bert=bert)
    reg_sbert = RegressionSentenceBERT(bert=bert)
    trip_sbert = TripletSentenceBERT(bert=bert)

    BATCH_SIZE = 4
    SEQ_LEN = 128
    sent1 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    sent2 = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    output = cls_sbert(sent1=sent1, sent2=sent2)
    output

    output = reg_sbert(sent1=sent1, sent2=sent2)
    output

    a = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    p = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    n = torch.randint(low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
    output = trip_sbert(anchor=a, pos=p, neg=n)
    output.shape
    output