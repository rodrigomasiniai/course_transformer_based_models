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
    def __init__(self, data_dir, tokenizer, max_len):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]")

        self.corpus = self._prepare_corpus(data_dir=data_dir, tokenizer=tokenizer)
        self.nsp_data = self._prepare_nsp_data(corpus=self.corpus)

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
        return len(self.nsp_data)

    def __getitem__(self, idx):
        dic = self.nsp_data[idx]
        x = torch.as_tensor((
            [self.cls_id]\
                + dic["segment1_ids"]\
                + [self.sep_id]\
                + dic["segment2_ids"]\
                + [self.sep_id]\
                + [self.pad_id] * (self.max_len - len(dic["segment1_ids"]) + len(dic["segment2_ids"]) - 3)\
        )[: self.max_len])
        return x
 

if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)

    data_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"

    MAX_LEN = 512
    ds = BookCorpusForBERT(data_dir=data_dir, tokenizer=tokenizer, max_len=MAX_LEN)
    BATCH_SIZE = 8
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, data in enumerate(dl, start=1):
        data == 2
        # print(
        #     data["gt_tokens"].shape,
        #     data["replaced_tokens"].shape,
        #     data["target_ids"].shape,
        #     data["segment_indices"].shape,
        #     data["is_next"].shape
        # )
