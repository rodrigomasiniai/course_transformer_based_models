import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert.model import BERTBase
from sentence_bert.model import SentenceBERTForRegression
from sentence_bert.stsb import STSbenchmarkDataset, STSbenchmarkCollator

REG_LOSS_FN = nn.MSELoss()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    bert_base = BERTBase(vocab_size=tokenizer.vocab_size)
    reg_sbert = SentenceBERTForRegression(embedder=bert_base)

    stsb_ds = STSbenchmarkDataset(
        csv_path="/Users/jongbeomkim/Documents/datasets/stsbenchmark/sts-train.csv", tokenizer=tokenizer
    )
    stsb_collator = STSbenchmarkCollator(tokenizer=tokenizer)
    stsb_dl = DataLoader(
        stsb_ds,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        collate_fn=stsb_collator
    )
    for batch, (score, sent1, sent2) in enumerate(stsb_dl, start=1):
        logit = reg_sbert(sent1=sent1, sent2=sent2)
        loss = REG_LOSS_FN(logit, score)
        loss
