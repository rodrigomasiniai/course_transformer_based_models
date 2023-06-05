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
import pysbd

# 원래의 RoBERTa는 BPE를 사용함
from bert.tokenize import prepare_bert_tokenizer

np.set_printoptions(edgeitems=20, linewidth=sys.maxsize, suppress=False)
torch.set_printoptions(edgeitems=16, linewidth=sys.maxsize, sci_mode=True)

SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def _disambiguate_sentence_boundary(text, segmenter=SEGMENTER):
    segmented = segmenter.segment(text)
    return [i.strip() for i in segmented]


class BookCorpusForRoBERTa(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer,
        max_len,
        mode: Literal["segment_pair", "sentence_pair", "full_sentences", "doc_sentences",]="doc_sentences"
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")

        self.corpus = self._prepare_corpus(data_dir=data_dir, tokenizer=tokenizer)
        self.data = self._prepare_data(self.corpus)
    
    def _prepare_corpus(self, data_dir, tokenizer):
        corpus = list()
        for doc_path in tqdm(list(Path(data_dir).glob("**/*.txt"))):
            for seg in open(doc_path, mode="r", encoding="utf-8"):
                seg = seg.strip()
                if seg == "":
                    continue

                sents = _disambiguate_sentence_boundary(seg)
                for sent in sents:
                    token_ids = tokenizer.encode(sent).ids
                    corpus.append({"document": str(doc_path), "segment": seg, "sentence": sent, "token_indices": token_ids})
        return corpus

    def _merge_ids_and_add_special_tokens_1(self, ids):
        merged = (
            [self.cls_id] + ids[0][: self.max_len - 3] + [self.sep_id] + ids[1]
        )[: self.max_len - 1] + [self.sep_id]
        merged += [self.pad_id] * (self.max_len - len(merged))
        return merged

    def _merge_ids_and_add_special_tokens_2(self, ids):
        merged = sum(ids, list())
        merged = merged[: self.max_len - 2]
        merged = [self.cls_id] + merged + [self.sep_id]
        merged += [self.pad_id] * (self.max_len - len(merged))
        return merged

    def _prepare_data(self, corpus):
        data = list()

        # "Each input has a pair of segments, which can each contain multiple natural sentences,
        # but the total combined length must be less than 512 tokens."
        if self.mode == "segment_pair":
            for id1 in range(len(corpus) - 1):
                if random.random() < 0.5:
                    is_next = True
                    id2 = id1 + 1
                else:
                    is_next = False
                    id2 = random.randrange(len(corpus))
                seg = [corpus[id1]["segment"], corpus[id2]["segment"]]
                ids = [corpus[id1]["token_indices"], corpus[id2]["token_indices"]]

                merged = self._merge_ids_and_add_special_tokens_1(ids)
                data.append({"text": seg, "token_indices": ids, "merged_ids": merged, "is_next": is_next})

        # Each input contains a pair of natural sentences,
        # either sampled from a contiguous portion of one document or from separate documents.
        # Since these inputs are significantly shorter than 512 tokens, we increase the batch size so that the total number of tokens remains similar to "SEGMENT-PAIR" + NSP.
        if self.mode == "sentence_pair":
            for id1 in range(len(corpus) - 1):
                if random.random() < 0.5:
                    is_next = True
                    id2 = id1 + 1
                else:
                    is_next = False
                    id2 = random.randrange(len(corpus))
                sents = [corpus[id1]["sentence"], corpus[id2]["sentence"]]
                token_ids_ls = [corpus[id1]["token_indices"], corpus[id2]["token_indices"]]

                token_ids = self._merge_ids_and_add_special_tokens_1(token_ids_ls)
                data.append({"sentences": sents, "list_of_token_indices": token_ids_ls, "token_indices": token_ids, "is_next": is_next})

        elif self.mode == "full_sentences":
            text = [corpus[0]["sentence"]]
            ids = [corpus[0]["token_indices"]]
            for id_ in range(1, len(corpus)):
                # "Inputs may cross document boundaries. When we reach the end of one document,
                # we begin sampling sentences from the next document
                # and add an extra separator token between documents."
                if corpus[id_ - 1]["document"] != corpus[id_]["document"]:
                    ids.append([self.sep_id])

                # Each input is packed with full sentences sampled contiguously
                # from one or more documents, such that the total length is at most 512 tokens.
                if len(sum(ids, list())) + len(corpus[id_]["token_indices"]) > self.max_len - 2 or\
                id_ == len(corpus) - 1:
                    merged = self._merge_ids_and_add_special_tokens_2(ids)
                    data.append({"text": text, "token_indices": ids, "merged_ids": merged})

                    text = list()
                    ids = list()
                text.append(corpus[id_]["text"])
                ids.append(corpus[id_]["token_indices"])

        # "Inputs sampled near the end of a document may be shorter than 512 tokens,
        # so we dynamically increase the batch size in these cases
        # to achieve a similar number of total tokens as 'FULL-SENTENCES'."
        # 어떻게 구현할 것인가?
        elif self.mode == "doc_sentences":
            text = [corpus[0]["sentence"]]
            ids = [corpus[0]["token_indices"]]
            for id_ in range(1, len(corpus)):
                # except that they may not cross document boundaries.
                # Inputs are constructed similarly to "FULL-SENTENCES",
                if corpus[id_ - 1]["document"] != corpus[id_]["document"] or\
                len(sum(ids, list())) + len(corpus[id_]["token_indices"]) > self.max_len - 2 or\
                id_ == len(corpus) - 1:
                    merged = self._merge_ids_and_add_special_tokens_2(ids)
                    data.append({"text": text, "token_indices": ids, "merged_ids": merged})

                    text = list()
                    ids = list()
                text.append(corpus[id_]["text"])
                ids.append(corpus[id_]["token_indices"])
        return data

    def _get_segment_indices_from_token_indices(self, token_ids):
        seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
        is_sep = (token_ids == self.sep_id)
        if is_sep.sum() == 2:
            a, b = is_sep.nonzero()
            seg_ids[a + 1: b + 1] = 1
        return seg_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode in ["segment_pair", "sentence_pair"]:
            token_ids = torch.as_tensor(self.data[idx]["token_indices"])
            seg_ids = self._get_segment_indices_from_token_indices(token_ids)
            return token_ids, seg_ids, torch.as_tensor(self.data[idx]["is_next"])
        else:
            return torch.as_tensor(self.data[idx]["merged_ids"])


if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
    data_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"
    MAX_LEN = 512
    ds = BookCorpusForRoBERTa(tokenizer=tokenizer, data_dir=data_dir, max_len=MAX_LEN, mode="sentence_pair")
    ds.corpus[12]
    BATCH_SIZE = 8
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # for batch, token_ids in enumerate(dl, start=1):
    for batch, (token_ids, seg_ids, is_next) in enumerate(dl, start=1):
        token_ids
