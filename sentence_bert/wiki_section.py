# References
    # https://github.com/sebastianarnold/WikiSection.git
    # https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/other/training_wikipedia_sections.py

import sys
import torch
from torch.utils.data import DataLoader
import json
from fastapi.encoders import jsonable_encoder
from collections import defaultdict
from tqdm.auto import tqdm
import random
from copy import deepcopy

from bert.tokenize import prepare_bert_tokenizer

torch.set_printoptions(edgeitems=16, linewidth=sys.maxsize, sci_mode=True)


def _group_by_sections(wiki_data, tokenizer):
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    sents_by_sec = defaultdict(list)
    for block in tqdm(wiki_data):
        annot = block["annotations"]
        parag = block["text"]
        for section in annot:
            sent = parag[section["begin"]: section["begin"] + section["length"]]
            token_ids = [cls_id] + tokenizer.encode(sent).ids + [sep_id]
            sents_by_sec[section["sectionLabel"]].append(token_ids)
    return sents_by_sec


def _sample_positive_sentence(sents_by_sec, anchor_sec, anchor):
    sents = sents_by_sec[anchor_sec]
    pos = deepcopy(anchor)
    while pos == anchor:
        pos = random.choice(sents)
    return pos


def _sample_negative_sentence(sents_by_sec, anchor_sec):
    secs = list(sents_by_sec)
    neg_sec = deepcopy(anchor_sec)
    while neg_sec == anchor_sec:
        neg_sec = random.choice(secs)
    neg = random.choice(sents_by_sec[neg_sec])
    return neg


def _get_wiki_section_dataset(json_path, tokenizer):
    with open(json_path, mode="r") as f:
        wiki_data = jsonable_encoder(json.load(f))
        sents_by_sec = _group_by_sections(wiki_data, tokenizer=tokenizer)

    ds = list()
    for anchor_sec in sents_by_sec.keys():
        for anchor in sents_by_sec[anchor_sec]:
            pos = _sample_positive_sentence(sents_by_sec=sents_by_sec, anchor_sec=anchor_sec, anchor=anchor)
            neg = _sample_negative_sentence(sents_by_sec=sents_by_sec, anchor_sec=anchor_sec)

            ds.append((anchor, pos, neg))
    return ds


class WikiSectionCollator(object):
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
        anchor_max_len = min(self.max_len, max([len(anchor) for anchor, _, _ in batch]))
        pos_max_len = min(self.max_len, max([len(pos) for _, pos, _ in batch]))
        neg_max_len = min(self.max_len, max([len(neg) for _, _, neg in batch]))

        anchors, poss, negs = list(), list(), list()
        for anchor, pos, neg in batch:
            anchor = self._truncate_or_pad(token_ids=anchor, max_len=anchor_max_len)
            pos = self._truncate_or_pad(token_ids=pos, max_len=pos_max_len)
            neg = self._truncate_or_pad(token_ids=neg, max_len=neg_max_len)

            anchors.append(anchor)
            poss.append(pos)
            negs.append(neg)
        return torch.as_tensor(anchors), torch.as_tensor(poss), torch.as_tensor(negs)


if __name__ == "__main__":
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)

    json_path = "/Users/jongbeomkim/Documents/datasets/wikisection_dataset_json/wikisection_en_city_train.json"
    ds = _get_wiki_section_dataset(json_path=json_path, tokenizer=tokenizer)
    MAX_LEN = 512
    collator = WikiSectionCollator(tokenizer=tokenizer, max_len=MAX_LEN)
    BATCH_SIZE = 8
    dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collator)
    for batch, (anchor, pos, neg) in enumerate(dl, start=1):
        print(anchor.shape, pos.shape, neg.shape)
