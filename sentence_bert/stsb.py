import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

MAX_LEN = 512


class STSbenchmarkDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=MAX_LEN):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.scores, sents1, sents2 = list(), list(), list()
        with open(csv_path, mode="r") as f:
            for line in f:
                line = line.split("\t")
                if len(line) == 7:
                    line = line[4:]
                elif len(line) == 9:
                    line = line[4: -2]
                score, sent1, sent2 = line
                score = float(score)
                score = self._normalize_score(score)

                self.scores.append(score)
                sents1.append(sent1)
                sents2.append(sent2)
                
        self.ids1 = torch.as_tensor(tokenizer(sents1, padding="max_length", max_length=max_len)["input_ids"])
        self.ids2 = torch.as_tensor(tokenizer(sents2, padding="max_length", max_length=max_len)["input_ids"])

        # "We implemented a smart batching strategy: Sentences with similar lengths are grouped together"
        # in the section 7 of the paper.
        order = torch.argsort(self._get_length(self.ids1) + self._get_length(self.ids1))
        self.ids1 = self.ids1[order]
        self.ids2 = self.ids2[order]

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.scores[idx], self.ids1[idx], self.ids2[idx]

    def _normalize_score(self, score):
        score -= 2.5
        score /= 2.5
        return score

    def _get_length(self, x):
        return (x != 0).sum(dim=1)


def _truncate_to_max_length(batch, tokenizer):
    scores, sents1, sents2 = list(), list(), list()
    for score, sent1, sent2 in batch:
        scores.append(score)
        sents1.append(sent1)
        sents2.append(sent2)

    scores, sents1, sents2 = torch.as_tensor(scores), torch.stack(sents1), torch.stack(sents2)
    # "and are only padded to the longest element in a mini-batch. This drastically reduces
    # computational overhead from padding tokens."
    sents1, sents2 = (
        sents1[:, : (sents1 != tokenizer.pad_token_id).sum(dim=1).max()],
        sents2[:, : (sents2 != tokenizer.pad_token_id).sum(dim=1).max()]
    )
    return scores, sents1, sents2


if __name__ == "__main__":
    csv_path = "/Users/jongbeomkim/Documents/datasets/stsbenchmark/sts-train.csv"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    stsb_ds = STSbenchmarkDataset(csv_path=csv_path, tokenizer=tokenizer)
    stsb_dl = DataLoader(
        stsb_ds,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        collate_fn=_truncate_to_max_length(tokenizer=tokenizer)
    )
    for batch, (score, sent1, sent2) in enumerate(stsb_dl, start=1):
        print(sent1.shape, sent2.shape)
