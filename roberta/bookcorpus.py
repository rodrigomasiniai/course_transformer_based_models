import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm

from bert.tokenize import prepare_bert_tokenizer


class RoBERTaDataset(Dataset):
    def __init__(self, tokenizer, corpus_dir, seq_len):
        self.tokenizer = tokenizer
        self.corpus_dir = corpus_dir
        self.seq_len = seq_len

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")

        self.corpus = self._collect_corpus(corpus_dir=corpus_dir, seq_len=seq_len)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus[idx]

    def _collect_corpus(self, corpus_dir, seq_len):
        corpus = list()
        for doc_path in tqdm(list(Path(corpus_dir).glob("**/*.txt"))):
            temp = [self.cls_id]
            for line in open(doc_path, mode="r", encoding="utf-8"):
                if line in ["\n"]:
                    continue
                # line.replace("\n", "").replace("\t", "")
                encoded = tokenizer.encode(line)
                ids = encoded.ids
                if len(temp) + len(ids) <= seq_len - 2:
                    temp.extend(ids)
                else:
                    temp.extend([self.pad_id] * (seq_len - len(temp) - 1) + [self.sep_id])
                    corpus.append(temp)

                    temp = [self.cls_id]

        corpus = torch.as_tensor(corpus, dtype=torch.int64)
        return corpus


if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
    corpus_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"
    SEQ_LEN = 512
    ds = RoBERTaDataset(tokenizer=tokenizer, corpus_dir=corpus_dir, seq_len=SEQ_LEN)
    BATCH_SIZE = 8
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    for batch, data in enumerate(dl, start=1):
        data.shape
