# References
    # https://rowanzellers.com/swag/
    # https://github.com/rowanz/swagaf/tree/master/data

# "Given a partial description like 'she opened the hood of the car,' humans can reason about the situation
# and anticipate what might come next ('then, she examined the engine').
# SWAG (Situations With Adversarial Generations) is a large-scale dataset for this task of grounded commonsense inference,
# unifying natural language inference and physically grounded reasoning.
# Each question is a video caption from LSMDC or ActivityNet Captions, with four answer choices
# about what might happen next in the scene. The correct answer is the (real) video caption for the next event in the video;
# the three incorrect answers are adversarially generated and human verified, so as to fool machines but not humans.

import sys
import torch
import torch.nn as nn
import pandas as pd

from bert.tokenize import prepare_bert_tokenizer
from bert.model import BERTBase, SingleSequenceChoiceHead

pd.options.display.width = sys.maxsize
pd.options.display.max_columns = 10

# "We construct four input sequences, each containing the concatenation of the given sentence (sentence A)
# and a possible continuation (sentence B). The only task-specific parameters introduced is a vector
# whose dot product with the [CLS] token representation $C$ denotes a score for each choice
# which is normalized with a softmax layer."
# We fine-tune the model for 3 epochs with a learning rate of 2e-5 and a batch size of 16.
N_EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 16


cls_id = tokenizer.token_to_id("[CLS]")
sep_id = tokenizer.token_to_id("[SEP]")
pad_id = tokenizer.token_to_id("[PAD]")
unk_id = tokenizer.token_to_id("[UNK]")

def _preprocess_raw_data(csv_path):
    raw_data = pd.read_csv(csv_path)
    for col in ["ending0", "ending1", "ending2", "ending3"]:
        raw_data[col] = raw_data.apply(lambda x: f"""{x["sent2"]} {x[col]}""", axis=1)
    raw_data = raw_data[["sent1", "ending0", "ending1", "ending2", "ending3", "label"]]
    return raw_data
csv_path = "/Users/jongbeomkim/Documents/datasets/swag/train.csv"
corpus = _preprocess_raw_data(csv_path)

vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)

sent2encoded = {sent: tokenizer.encode(sent) for sent in corpus["sent1"].unique()}

for row in corpus.itertuples():
    token_ids = [cls_id] + sent2encoded[row.sent1].ids + [sep_id] + tokenizer.encode(row.ending0).ids + [sep_id]
    (8, 4, 512)
    