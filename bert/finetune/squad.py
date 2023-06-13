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
import torch.nn.functional as F
import json
from fastapi.encoders import jsonable_encoder
# from pprint import pprint
from tqdm.auto import tqdm

from bert.tokenize import prepare_bert_tokenizer

torch.set_printoptions(precision=2, edgeitems=12, linewidth=sys.maxsize, sci_mode=True)


def parse_squad(json_path):
    with open(json_path, mode="r") as f:
        raw_data = jsonable_encoder(json.load(f))

    data = list()
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
                else:
                    start_id = end_id = 0
                data.append(
                    {"question": que, "context":ctx, "answer": {"start_index": start_id, "end_index": end_id}}
                )
    print(f"""There were {error_cnt} erro(s).""")
    return data

MAX_LEN = 512
class SQuADForBERT(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        stride=None,
        max_len=MAX_LEN,
    ):
        if stride is None:
            stride = max_len // 2
        
        cls_id = tokenizer.token_to_id("[CLS]")
        sep_id = tokenizer.token_to_id("[SEP]")
        pad_id = tokenizer.token_to_id("[PAD]")
        unk_id = tokenizer.token_to_id("[UNK]")
        
        json_path = "/Users/jongbeomkim/Documents/datasets/train-v2.0.json"
        data = parse_squad(json_path)

        new_data = list()
        for line in data:
            que_token_ids = tokenizer.encode(line["question"]).ids
            ctx_encoded = tokenizer.encode(line["context"])
            ctx_token_ids = ctx_encoded.ids

            if (line["answer"]["start_index"], line["answer"]["end_index"]) == (0, 0):
                (start_id, end_id) = (0, 0)
            else:
                ctx_offsets = ctx_encoded.offsets

                for id_, (s, e) in enumerate(ctx_offsets):
                    if s <= line["answer"]["start_index"] < e:
                        # start_id = id_ + len_que + 2
                        start_id = id_
                    if s < line["answer"]["end_index"] <= e:
                        # end_id = id_ + len_que + 3
                        end_id = id_ + 1

            len_que = len(que_token_ids)
            len_ctx = len(ctx_token_ids)
            start_id += len_que + 2
            end_id += len_que + 2

            if len_que + len_ctx + 3 > max_len:
                for i in range(0, len_ctx // 2, stride):
                    token_ids = [cls_id] + que_token_ids + [sep_id] + ctx_token_ids[i: i + max_len - len_que - 3] + [sep_id]
                    if not (i <= start_id or end_id < i + max_len - len_que - 3):
                        (start_id, end_id) = (0, 0)
                    else:
                        start_id -= i
                        end_id -= i

                    new_data.append({"token_indices": token_ids, "start_index": start_id, "end_index": end_id})
            else:
                token_ids = [cls_id] + que_token_ids + [sep_id] + ctx_token_ids + [sep_id]
                # tokenizer.decode(token_ids[start_id: end_id]), line["context"][line["answer"]["start_index"]: line["answer"]["end_index"]]

                new_data.append({"token_indices": token_ids, "start_index": start_id, "end_index": end_id})

        for i in range(10):
            tokenizer.decode(new_data[i]["token_indices"][new_data[i]["start_index"]: new_data[i]["end_index"]])
        # for i in range(30):
        i = 1
        data[i]["context"][data[i]["answer"]["start_index"]: data[0]["answer"]["end_index"]]





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
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path, corpus_files=corpus_files)