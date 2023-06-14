# References
    # https://rajpurkar.github.io/SQuAD-explorer/
    # https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    # https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/
    # https://huggingface.co/learn/nlp-course/chapter7/7?fw=tf

# we represent the input question and passage as a single packed sequence,
# with the question using the A embedding and the passage using the B embedding.

# SQuAD 1.1, the previous version of the SQuAD dataset,
# contains 100,000+ question-answer pairs on 500+ articles.
# SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions
# written adversarially by crowdworkers to look similar to answerable ones.

import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
from fastapi.encoders import jsonable_encoder
from tqdm.auto import tqdm

from bert.tokenize import prepare_bert_tokenizer

torch.set_printoptions(precision=2, edgeitems=12, linewidth=sys.maxsize, sci_mode=True)

MAX_LEN = 512


class SQuADForBERT(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        max_len=MAX_LEN,
        stride=None,
    ):
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride

        if self.stride is None:
            self.stride = self.max_len // 2
        
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.unk_id = tokenizer.token_to_id("[UNK]")
        
        self.corpus = self._prepare_corpus(json_path)
        self.data = self._prepare_data(self.corpus)

    def _prepare_corpus(self, json_path):
        with open(json_path, mode="r") as f:
            raw_data = jsonable_encoder(json.load(f))

        corpus = list()
        error_cnt = 0
        for article in raw_data["data"]:
            for parags in article["paragraphs"]:
                ctx = parags["context"]
                for qa in parags["qas"]:
                    que = qa["question"]
                    ans = qa["answers"]
                    if ans:
                        start_id = ans[0]["answer_start"]
                        end_id = start_id + len(ans[0]["text"])
                        if ctx[start_id: end_id] != ans[0]["text"]:
                            error_cnt += 1
                            continue
                        if "split with " in ans[0]["text"]:
                            start_id, end_id, ctx[start_id: end_id], ctx[start_id: end_id + 5], ans[0]
                    else:
                        start_id = end_id = 0
                    corpus.append(
                        {"question": que, "context":ctx, "answer": {"start_index": start_id, "end_index": end_id}}
                    )
        print(f"""There were {error_cnt} erro(s).""")
        return corpus

    def _prepare_data(self, corpus):
        data = list()
        for line in tqdm(corpus):
            que_token_ids = tokenizer.encode(line["question"]).ids
            ctx_encoded = tokenizer.encode(line["context"])
            ctx_token_ids = ctx_encoded.ids

            if (line["answer"]["start_index"], line["answer"]["end_index"]) == (0, 0):
                (start_id, end_id) = (0, 0)
            else:
                ctx_offsets = ctx_encoded.offsets

                for id_, (s, e) in enumerate(ctx_offsets):
                    if s <= line["answer"]["start_index"] < e:
                        start_id = id_
                    if s < line["answer"]["end_index"] <= e:
                        end_id = id_ + 1

            len_que = len(que_token_ids)
            len_ctx = len(ctx_token_ids)
            start_id += len_que + 2
            end_id += len_que + 2

            if len_que + len_ctx + 3 > self.max_len:
                for i in range(0, len_ctx // 2, self.stride):
                    token_ids = [self.cls_id] + que_token_ids + [self.sep_id] + ctx_token_ids[i: i + self.max_len - len_que - 3] + [self.sep_id]
                    if not (i <= start_id and end_id < i + self.max_len - len_que - 3):
                        start_id, end_id = 0, 0
                    else:
                        start_id -= i
                        end_id -= i

                    data.append({"token_indices": token_ids, "start_index": start_id, "end_index": end_id})
            else:
                token_ids = [self.cls_id] + que_token_ids + [self.sep_id] + ctx_token_ids + [self.sep_id]

                data.append({"token_indices": token_ids, "start_index": start_id, "end_index": end_id})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

json_path = "/Users/jongbeomkim/Documents/datasets/train-v2.0.json"
vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
squad_ds = SQuADForBERT(json_path=json_path, tokenizer=tokenizer, max_len=MAX_LEN)
len(squad_ds)
squad_ds.data

class QuestionAnsweringHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        # "We only introduce a start vector $S \in \mathbb{R}^{H}$ and an end vector
        # $E \in \mathbb{R}^{H}$ during fine-tuning."
        self.proj = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # "The probability of word $i$ being the start of the answer span is computed
        # as a dot product between $T_{i}$ and $S$ followed by a softmax over all of the words in the paragraph."
        x = self.proj(x)
        start_logit, end_logit = torch.split(x, split_size_or_sections=1, dim=2)
        start_logit, end_logit = start_logit.squeeze(), end_logit.squeeze()
        start_id, end_id = torch.argmax(start_logit, dim=1), torch.argmax(end_logit, dim=1)
        return start_id, end_id
head = QuestionAnsweringHead(hidden_dim=768)
x = torch.randn((8, 512, 768))
head(x)

# from transformers import BertForQuestionAnswering

# The training objective is the sum of the log-likelihoods of the correct start and end positions.
# We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32. Table 2 shows top leaderboard entries as well

if __name__ == "__main__":