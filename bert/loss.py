import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert.model import BERT, MaskedLanguageModelHead, NextSentencePredictionHead
from bert.data import BERTDataset


class BERTLoss(nn.Module):
    def __init__(self, lamb=3000):
        super().__init__()
        self.lamb = lamb

    def forward(self, mlm_logits, nsp_logits, labels):
        raw_mlm_loss = F.cross_entropy(mlm_logits.permute(0, 2, 1), labels["ground_truth_ids"], reduction="none")
        raw_mlm_loss *= labels["prediction_target"]
        mlm_loss = raw_mlm_loss.sum()

        nsp_loss = F.cross_entropy(nsp_logits, labels["is_next"])
        # `mlm_loss`와 `nsp_loss` 사이의 Imbalance??
        loss = mlm_loss + self.lamb * nsp_loss
        return mlm_loss, nsp_loss


if __name__ == "__main__":
    VOCAB_SIZE = 30_522
    bert = BERT(vocab_size=VOCAB_SIZE)
    mlm_head = MaskedLanguageModelHead(vocab_size=VOCAB_SIZE)
    nsp_head = NextSentencePredictionHead()
    
    criterion = BERTLoss()

    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab.json"
    corpus_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"

    SEQ_LEN = 512
    # BATCH_SIZE = 256
    BATCH_SIZE = 8
    ds = BERTDataset(vocab_path=vocab_path, corpus_dir=corpus_dir, seq_len=SEQ_LEN)
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, data in enumerate(dl, start=1):
        bert_output = bert(seq=data["masked_ids"], seg_label=data["segment_label"])
        mlm_logits = mlm_head(bert_output)
        nsp_logits = nsp_head(bert_output)
        
        criterion(mlm_logits=mlm_logits, nsp_logits=nsp_logits, labels=data)
        
        bert_output.shape, mlm_logits.shape, nsp_logits.shape

    N_STEPS = 1_000_000
    N_WARMUP_STEPS = 10_000
    MAX_LR = 1e-4