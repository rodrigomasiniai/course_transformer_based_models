import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.linalg as LA

from bert.tokenize import prepare_bert_tokenizer
from bert.model import BERTBase
from sentence_bert.model import SentenceBERTForRegression, SentenceBERTWithTripletNetworks
from sentence_bert.sts_benchmark import STSbenchmarkDataset, STSbenchmarkCollator
from sentence_bert.wiki_section import _get_wiki_section_dataset, WikiSectionCollator

VOCAB_SIZE = 30_522
MAX_LEN = 512
BATCH_SIZE = 8

CLS_LOSS_FN = nn.CrossEntropyLoss()
REG_LOSS_FN = nn.MSELoss()


class TripletLossFunction(nn.Module):
    def __init__(self, eps=1):
        super().__init__()

        self.eps = eps

    def forward(self, a, p, n):
        x = LA.vector_norm(a - p, ord=2, dim=1) - LA.vector_norm(a - n, ord=2, dim=1) + self.eps
        x = F.relu(x)
        return x


TRIP_LOSS_FN = TripletLossFunction()


if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)

    bert_base = BERTBase(vocab_size=VOCAB_SIZE)
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


    trip_sbert = SentenceBERTWithTripletNetworks(embedder=bert_base)
    json_path = "/Users/jongbeomkim/Documents/datasets/wikisection_dataset_json/wikisection_en_city_train.json"

    wiki_ds = _get_wiki_section_dataset(json_path=json_path, tokenizer=tokenizer)
    wiki_collator = WikiSectionCollator(tokenizer=tokenizer, max_len=MAX_LEN)
    wiki_dl = DataLoader(
        dataset=wiki_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=wiki_collator
    )
    for batch, (a, p, n) in enumerate(wiki_dl, start=1):
        logit = trip_sbert(a=a, p=p, n=n)
        loss = TRIP_LOSS_FN(*logit)
        loss
