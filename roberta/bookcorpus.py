import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from typing import Literal
import random

# 원래의 RoBERTa는 BPE를 사용함
from bert.tokenize import prepare_bert_tokenizer


class RoBERTaDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        corpus_dir,
        seq_len,
        mode: Literal["segment_pair", "sentence_pair", "full_sentences", "doc_sentences",]="doc_sentences"
    ):
        self.tokenizer = tokenizer
        self.corpus_dir = corpus_dir
        self.seq_len = seq_len
        self.mode = mode

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")

        self.corpus = self._collect_corpus(corpus_dir=corpus_dir, seq_len=seq_len, mode=mode)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus[idx]

    def _collect_corpus(self, corpus_dir, seq_len, mode, shuffle_docs=True):
        docs = list(Path(corpus_dir).glob("**/*.txt"))
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
                    if len(seq) + len(ids) <= seq_len - 1: # "such that the total length is at most 512 tokens"
                        seq.extend(ids)
                    else:
                        seq.extend([self.pad_id] * (seq_len - len(seq) - 1) + [self.sep_id])
                        corpus.append(seq)

                        seq = [self.cls_id]
                if mode == "segment_pair":
                    corpus.append(ids)

        if mode == "segment_pair":
            random.shuffle(corpus)
            new_corpus = list()
            seq = [self.cls_id]
            for i in range(len(corpus) // 2):
                seq1 = corpus[2 * i]
                seq2 = corpus[2 * i + 1]
                if len(seq1) + len(seq1) - 3 > seq_len:


            corpus = [
                [self.cls_id] + corpus[2 * i] + [self.sep_id] +  corpus[2 * i + 1] + [self.sep_id]
                for i in range(len(corpus) // 2)
            ]
        corpus = torch.as_tensor(corpus, dtype=torch.int64)
        # Inputs sampled near the end of a document may be shorter than 512 tokens,
        # so we dynamically increase the batch size in these cases
        # to achieve a similar number of total tokens as "FULL-SENTENCES".
        return corpus


if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
    corpus_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"
    SEQ_LEN = 512
    ds = RoBERTaDataset(tokenizer=tokenizer, corpus_dir=corpus_dir, seq_len=SEQ_LEN, mode="doc_sentences")
    # ds = RoBERTaDataset(tokenizer=tokenizer, corpus_dir=corpus_dir, seq_len=SEQ_LEN, mode="full_sentences")
    BATCH_SIZE = 8
    # dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    for batch, data in enumerate(dl, start=1):
        if torch.isin(data, ds.sep_id).sum() != 8:
            seq = ((data == 2).numpy() * 255).astype("uint8")
            show_image(seq)
