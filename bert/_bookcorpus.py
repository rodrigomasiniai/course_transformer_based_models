# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html#sec-bert-dataset
    # https://nn.labml.ai/transformers/mlm/index.html

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import json
from pathlib import Path
from tqdm.auto import tqdm

# from bert.wordpiece_implementation import collect_corpus, tokenize
from bert.tokenize import prepare_bert_tokenizer
from bert.model import BERTBase


class BookCorpusForBERT(Dataset):
    def __init__(self, vocab_path, data_dir, max_len):
        self.vocab_path = vocab_path
        self.data_dir = data_dir
        self.max_len = max_len

        with open(vocab_path, mode="r") as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)

        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]

        self.id_to_token = {v:k for k, v in self.vocab.items()}

        self.corpus = collect_corpus(data_dir=data_dir, add_empty_string=True)
      
    def _mask_some_tokens(self, tokens):
        gt_tokens, replaced_tokens, target_ids = list(), list(), list()
        for id_, token in enumerate(tokens):
            if id_ >= self.max_len:
                continue
            id_ = self.vocab.get(token, self.unk_id)
            gt_tokens.append(id_)

            rand_var = random.random()
            if rand_var < 0.15:
                target_ids.append(True)

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
                target_ids.append(False)
            replaced_tokens.append(id_)

        # Pad.
        gt_tokens.extend([self.pad_id] * (self.max_len - len(gt_tokens)))
        replaced_tokens.extend([self.pad_id] * (self.max_len - len(replaced_tokens)))
        target_ids.extend([self.pad_id] * (self.max_len - len(target_ids)))

        gt_tokens = torch.as_tensor(gt_tokens, dtype=torch.int64)
        replaced_tokens = torch.as_tensor(replaced_tokens, dtype=torch.int64)
        target_ids = torch.as_tensor(target_ids, dtype=torch.bool)
        output = {"gt_tokens": gt_tokens, "replaced_tokens": replaced_tokens, "target_ids": target_ids}
        return output

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

        # Truncate.
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2[: self.max_len - len(tokens1) - 3] + ["[SEP]"]
        seg_ids = ([1] * (len(tokens1) + 2) + [2] * (len(tokens2) + 1))[: self.max_len]

        # 첫 번째 문장: 0, 두 번째 문장: 1, Pad: 0
        seg_ids.extend([self.pad_id] * (self.max_len - len(seg_ids)))

        seg_ids = torch.as_tensor(seg_ids, dtype=torch.int64)
        is_next = torch.as_tensor(is_next, dtype=torch.int64)
        output = {"tokens": tokens, "segment_indices": seg_ids, "is_next": is_next}
        return output

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        output = self._sample_next_sentence(idx=idx)
        is_next = output["is_next"]
        seg_ids = output["segment_indices"]

        output = self._mask_some_tokens(tokens=output["tokens"])
        output["is_next"] = is_next
        output["segment_indices"] = seg_ids
        return output
