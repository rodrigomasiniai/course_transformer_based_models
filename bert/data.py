# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import json

from bert.wordpiece import collect_corpus, tokenize
from bert.model import BERT


class BERTDataset(Dataset):
    def __init__(self, vocab_path, corpus_dir, seq_len):
        self.vocab_path = vocab_path
        self.corpus_dir = corpus_dir
        self.seq_len = seq_len

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
        for id_, token in enumerate(tokens):
            if id_ >= self.seq_len:
                continue
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

        # Pad.
        gt_ids.extend([self.pad_id] * (self.seq_len - len(gt_ids)))
        masked_ids.extend([self.pad_id] * (self.seq_len - len(masked_ids)))
        pred_trg.extend([self.pad_id] * (self.seq_len - len(pred_trg)))

        gt_ids = torch.as_tensor(gt_ids, dtype=torch.int64)
        masked_ids = torch.as_tensor(masked_ids, dtype=torch.int64)
        pred_trg = torch.as_tensor(pred_trg, dtype=torch.bool)
        output = {"ground_truth_ids": gt_ids, "masked_ids": masked_ids, "prediction_target": pred_trg}
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

        # if len(tokens1) + len(tokens2) >= self.seq_len - 3:
        # Truncate.
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2[: self.seq_len - len(tokens1) - 3] + ["[SEP]"]
        seg_label = ([1] * (len(tokens1) + 2) + [2] * (len(tokens2) + 1))[: self.seq_len]

        seg_label.extend([self.pad_id] * (self.seq_len - len(seg_label)))

        seg_label = torch.as_tensor(seg_label, dtype=torch.int64)
        # is_next = torch.as_tensor(is_next, dtype=torch.bool)
        is_next = torch.as_tensor(is_next, dtype=torch.int64)
        output = {"tokens": tokens, "segment_label": seg_label, "is_next": is_next}
        return output

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        output = self._sample_next_sentence(idx=idx)
        is_next = output["is_next"]
        seg_label = output["segment_label"]

        output = self._mask_some_tokens(tokens=output["tokens"])
        output["is_next"] = is_next
        output["segment_label"] = seg_label
        return output
 

if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab.json"
    corpus_dir = "/Users/jongbeomkim/Documents/datasets/bookscorpus_subset"

    SEQ_LEN = 512
    BATCH_SIZE = 8
    ds = BERTDataset(vocab_path=vocab_path, corpus_dir=corpus_dir, seq_len=SEQ_LEN)
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, data in enumerate(dl, start=1):
        print(
            data["ground_truth_ids"].shape,
            data["masked_ids"].shape,
            data["prediction_target"].shape,
            data["segment_label"].shape,
            data["is_next"].shape
        )
