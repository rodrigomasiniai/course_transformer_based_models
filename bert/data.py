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
      
    def _mask_some_tokens(self, tokens):
        gt_ids, masked_ids, pred_trg = list(), list(), list()
        for token in tokens:
            id_ = self.vocab.get(token, self.unk_id)
            gt_ids.append(id_)

            rand_var = random.random()
            if rand_var < 0.15:
                pred_trg.append(True)

                if rand_var < 0.15 * 0.8:
                    id_ = self.mask_id
                elif rand_var < 0.15 * 0.9:
                    while True:
                        id_ = random.randrange(self.vocab_size)
                        if id_ not in [
                            self.pad_id, self.unk_id, self.cls_id, self.sep_id, self.mask_id
                        ]:
                            break
            else:
                pred_trg.append(False)
            masked_ids.append(id_)

        dic = {"ground_truth_ids": gt_ids, "masked_ids": masked_ids, "prediction_target": pred_trg}
        return dic

    def _sample_next_sentence(self, idx):
        idx1 = idx

        rand_var = random.random()
        if rand_var < 0.5:
            idx2 = idx1 + 1
            is_next = True
        else:
            idx2 = random.randrange(len(self))
            is_next = False

        sent1 = self.corpus[idx1]
        sent2 = self.corpus[idx2]
        if sent2 == "":
            is_next = False
        tokens1 = tokenize(text=sent1, vocab=self.vocab)
        tokens2 = tokenize(text=sent2, vocab=self.vocab)

        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
        dic = {"tokens": tokens, "is_next": is_next}
        return dic

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        dic = self._sample_next_sentence(idx=idx)
        is_next = dic["is_next"]

        dic = self._mask_some_tokens(tokens=dic["tokens"])
        dic["is_next"] = is_next
        return dic


if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab.json"
    corpus_dir = "/Users/jongbeomkim/Documents/datasets/bookscorpus_subset"

    ds = BERTDataset(vocab_path=vocab_path, corpus_dir=corpus_dir)
    # ds.cls_id
    # vocab = ds.vocab
    # vocab_size = ds.vocab_size
    # unk_id = ds.unk_id
    # mask_id = ds.mask_id

    idx = 3
    dic = ds[idx]
    print(dic["ground_truth_ids"])
    print(dic["masked_ids"])
    print(dic["prediction_target"])
    print(dic["is_next"])

    SEQ_LEN = 512
    BATCH_SIZE = 256
    N_STEPS = 1_000_000
    N_WARMUP_STEPS = 10_000
    MAX_LR = 1e-4