import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bert.tokenize import prepare_bert_tokenizer

pd.options.display.max_colwidth = sys.maxsize
torch.set_printoptions(edgeitems=16, linewidth=sys.maxsize, sci_mode=True)


def _get_nli_dataset(txt_path, tokenizer):
    df = pd.read_csv(txt_path, sep="\t", keep_default_na=False, on_bad_lines="skip", low_memory=False)

    df = df[["sentence1", "sentence2", "gold_label"]]
    df = df[df["gold_label"].isin(["neutral", "entailment", "contradiction"])]
    df["gold_label"] = df["gold_label"].map({"entailment": 0, "neutral": 1, "contradiction": 2})
    df["gold_label"].unique()

    prem2token_ids = {sent: tokenizer.encode(sent).ids for sent in df["sentence1"].unique()}
    hypo2token_ids = {sent: tokenizer.encode(sent).ids for sent in df["sentence2"].unique()}

    nli_ds = list(zip(
        df["sentence1"].map(prem2token_ids), df["sentence2"].map(hypo2token_ids), df["gold_label"]
    ))
    return nli_ds


class NLICollator(object):
    def __init__(self, tokenizer, max_len):
        self.max_len = max_len

        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def _truncate_or_pad(self, token_ids, max_len):
        if len(token_ids) <= max_len:
            token_ids = token_ids + [self.pad_id] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[: max_len - 1] + [self.sep_id]
        return token_ids
    
    def __call__(self, batch):
        p_max_len = min(self.max_len, max([len(p) for p, _, _ in batch]))
        h_max_len = min(self.max_len, max([len(h) for _, h, _ in batch]))

        ps, hs, labels = list(), list(), list()
        for p, h, label in batch:
            p = self._truncate_or_pad(token_ids=p, max_len=p_max_len)
            h = self._truncate_or_pad(token_ids=h, max_len=h_max_len)

            ps.append(p)
            hs.append(h)
            labels.append(label)
        return torch.as_tensor(ps), torch.as_tensor(hs), torch.as_tensor(labels)


# if __name__ == "__main__":
#     # txt_path = "/Users/jongbeomkim/Documents/datasets/nli_1.0/nli_1.0_train.txt"
#     txt_path = "/Users/jongbeomkim/Documents/datasets/multinli_1.0/multinli_1.0_train.txt"
#     nli_ds = _get_nli_dataset(txt_path)

#     MAX_LEN = 512
#     vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
#     tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
#     nli_collator = NLICollator(tokenizer=tokenizer, max_len=MAX_LEN)
#     BATCH_SIZE = 8
#     nli_dl = DataLoader(
#         dataset=nli_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=nli_collator
#     )
#     for batch, (p, h, label) in enumerate(nli_dl, start=1):
#         print(p.shape, h.shape, label)
