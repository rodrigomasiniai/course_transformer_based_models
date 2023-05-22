# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import json
from pathlib import Path
# import pickle as pk

from wordpiece import collect_corpus, tokenize


class BERTDataset(Dataset):
    def __init__(self, vocab_path, corpus_dir):
        # super().__init__()

        self.vocab_path = vocab_path
        self.corpus_dir = corpus_dir

        with open(vocab_path, mode="r") as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]

        self.id_to_token = {v:k for k, v in self.vocab.items()}

        self.corpus = collect_corpus(corpus_dir=corpus_dir, add_empty_string=True)
      
        
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, id_):
        sent = self.corpus[id_]
        tokens = tokenize(text=sent, vocab=self.vocab)

        ids, label = [self.cls_id], [self.cls_id]
        label_sent = ""
        for token in tokens:
            id_ = self.vocab.get(token, self.unk_id)
            label.append(id_)
            if random.random() < 0.15:
                sampled = random.random()
                if sampled < 0.8:
                    id_ = self.mask_id
                elif sampled < 0.9:
                    id_ = random.randrange(self.vocab_size)
            ids.append(id_)
            label_sent += (" " + self.id_to_token[id_]).replace(" ##", "")
        ids += [self.sep_id]
        label += [self.sep_id]

        dic = {"sentence": sent, "ids": ids, "label": label, "label_sentence": label_sent}
        return dic

vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab.json"
corpus_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"
ds = BERTDataset(vocab_path=vocab_path, corpus_dir=corpus_dir)
id_ = 3
print(ds[id_])
print(ds[id_]["ids"])
print(ds[id_]["label"])
