# References
    # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html#sec-bert-dataset

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from typing import Literal
import random
import sys
import numpy as np

# 원래의 RoBERTa는 BPE를 사용함
from bert.tokenize import prepare_bert_tokenizer

np.set_printoptions(suppress=True, linewidth=sys.maxsize, threshold=sys.maxsize)
torch.set_printoptions(edgeitems=10, sci_mode=True)


mode="doc_sentences"
data = list()
text = [corpus[0]["text"]]
ids = [corpus[0]["ids"]]
for id_ in range(1, len(corpus)):
    if corpus[id_ - 1]["document"] != corpus[id_]["document"]:
        data.append({"text": text, "ids": ids})

        text = [corpus[id_]["text"]]
        ids = [corpus[id_]["ids"]]

    cur_len = sum([len(i) for i in ids])
    if cur_len + len(corpus[id_]["ids"]) <= max_len - 2:
        text.append(corpus[id_]["text"])
        ids.append(corpus[id_]["ids"])
data[0]["text"]
for k in range(100):
    sum([len(i) for i in data[k]["ids"]])


class BookCorpusForRoBERTa(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_len,
        mode: Literal["segment_pair", "sentence_pair", "full_sentences", "doc_sentences",]="doc_sentences"
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_len = max_len
        self.mode = mode

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")

        self.corpus = self._prepare_corpus(data_dir=data_dir, tokenizer=tokenizer)
    
    def _prepare_corpus(self, data_dir, tokenizer):
        corpus = list()
        for doc_path in tqdm(list(Path(data_dir).glob("**/*.txt"))):
            for para in open(doc_path, mode="r", encoding="utf-8"):
                para = para.strip()
                if para == "":
                    continue
                ids = tokenizer.encode(para).ids
                corpus.append({"document": str(doc_path), "text": para, "ids": ids})
        return corpus

    def _prepare_nsp_data(self, corpus):
        nsp_data = list()
        for id1 in range(len(corpus) - 1):
            seg1 = corpus[id1]["text"]
            seg1_ids = corpus[id1]["ids"]
            if random.random() < 0.5:
                seg2 = corpus[id1 + 1]["text"]
                seg2_ids = corpus[id1 + 1]["ids"]
                is_next = True
            else:
                id2 = random.randrange(len(corpus))
                seg2 = corpus[id2]["text"]
                seg2_ids = corpus[id2]["ids"]
                is_next = False
            nsp_data.append({
                "segment1": seg1,
                "segment2": seg2,
                "segment1_ids": seg1_ids,
                "segment2_ids": seg2_ids,
                "is_next": is_next,
            })
        return nsp_data

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus[idx]

    def _collect_corpus(self, data_dir, max_len, mode, shuffle_docs=True):
        docs = list(Path(data_dir).glob("**/*.txt"))
        if shuffle_docs:
            random.shuffle(docs)

        corpus = list()
        seq = [self.cls_id]
        for doc_id, doc_path in enumerate(tqdm(docs)):
            is_first_sent = True
            for line in open(doc_path, mode="r", encoding="utf-8"):
                line = line.strip()
                if line == "":
                    continue

                # "Inputs may cross document boundaries. When we reach the end of one document,
                # we begin sampling sentences from the next document
                # and add an extra separator token between documents."
                if mode == "full_sentences" and is_first_sent and doc_id != 0:
                    seq.append(self.sep_id)
                    is_first_sent = False

                encoded = tokenizer.encode(line)
                ids = encoded.ids
                if mode in ["full_sentences", "doc_sentences"]:
                    if len(seq) + len(ids) <= max_len - 1: # "such that the total length is at most 512 tokens"
                        seq.extend(ids)
                    else:
                        seq.extend([self.pad_id] * (max_len - len(seq) - 1) + [self.sep_id])
                        corpus.append(seq)

                        seq = [self.cls_id]
                if mode == "segment_pair":
                    corpus.append(ids)

        # if mode == "segment_pair":
        #     random.shuffle(corpus)
        #     new_corpus = list()
        #     seq = [self.cls_id]
        #     for i in range(len(corpus) // 2):
        #         seq1 = corpus[2 * i]
        #         seq2 = corpus[2 * i + 1]
        #         if len(seq1) + len(seq1) - 3 > max_len:


        #     corpus = [
        #         [self.cls_id] + corpus[2 * i] + [self.sep_id] +  corpus[2 * i + 1] + [self.sep_id]
        #         for i in range(len(corpus) // 2)
        #     ]
        corpus = torch.as_tensor(corpus, dtype=torch.int64)
        # Inputs sampled near the end of a document may be shorter than 512 tokens,
        # so we dynamically increase the batch size in these cases
        # to achieve a similar number of total tokens as "FULL-SENTENCES".
        return corpus


if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
    data_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"
    max_len = 512
    # ds = BookCorpusForRoBERTa(tokenizer=tokenizer, data_dir=data_dir, max_len=max_len, mode="doc_sentences")
    ds = BookCorpusForRoBERTa(tokenizer=tokenizer, data_dir=data_dir, max_len=max_len, mode="full_sentences")
    BATCH_SIZE = 8
    # dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    for batch, data in enumerate(dl, start=1):
        data
        data.shape
        # if torch.isin(data, ds.sep_id).sum() != 8:
        #     seq = ((data == 2).numpy() * 255).astype("uint8")
        #     show_image(seq)