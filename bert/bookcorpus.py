# References
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
    # # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html#sec-bert-dataset
    # https://nn.labml.ai/transformers/mlm/index.html

import sys
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

# np.set_printoptions(edgeitems=20, linewidth=sys.maxsize, suppress=False)
torch.set_printoptions(edgeitems=16, linewidth=sys.maxsize, sci_mode=True)


class BookCorpusForBERT(Dataset):
    def __init__(self, data_dir, tokenizer, max_len):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")

        self.corpus = self._prepare_corpus(data_dir=data_dir, tokenizer=tokenizer)
        self.data = self._prepare_data(corpus=self.corpus)

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

    def _prepare_data(self, corpus):
        random.shuffle(corpus)

        for id1 in range(len(corpus) - 1):
            if random.random() < 0.5:
                is_next = True
                id2 = id1 + 1
            else:
                is_next = False
                id2 = random.randrange(len(corpus))
            text = [corpus[id1]["text"], corpus[id2]["text"]]
            ids = [corpus[id1]["ids"], corpus[id2]["ids"]]

            merged = self._merge_ids_and_add_special_tokens_1(ids)
            data.append({"text": text, "ids": ids, "merged_ids": merged,"is_next": is_next})
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
        token_ids = torch.as_tensor(self.data[idx]["merged_ids"])
        seg_ids = self._get_segment_indices_from_token_indices(token_ids)
        return token_ids, seg_ids, torch.as_tensor(self.data[idx]["is_next"])


# def _get_segment_indices_from_token_indices(token_ids, sep_id):
#     seg_ids = torch.zeros_like(token_ids, dtype=token_ids.dtype, device=token_ids.device)
#     for i in range(token_ids.shape[0]):
#         is_sep = (token_ids[i] == sep_id)
#         if is_sep.sum() == 2:
#             a, b = is_sep.nonzero()
#             seg_ids[i, a + 1: b + 1] = 1
#     return seg_ids
 

if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)

    data_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"

    MAX_LEN = 512
    ds = BookCorpusForBERT(data_dir=data_dir, tokenizer=tokenizer, max_len=MAX_LEN)
    BATCH_SIZE = 4
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, data in enumerate(dl, start=1):
        data
