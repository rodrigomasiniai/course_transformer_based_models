# References
    # https://github.com/sebastianarnold/WikiSection.git
    # https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/other/training_wikipedia_sections.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import json
from fastapi.encoders import jsonable_encoder
from pprint import pprint
from collections import defaultdict
from itertools import permutations, combinations
from tqdm.auto import tqdm

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# df = pd.read_csv("/Users/jongbeomkim/Downloads/wikipedia-sections-triplets/test.csv")
# df.sort_values(by="Sentence1", inplace=True)
# df.head(10)
# len(df)

# df.loc[0, "Sentence1"], df.loc[0, "Sentence2"], df.loc[0, "Sentence3"]
# df.loc[1, "Sentence1"]


def _group_by_sections(dic):
    sec2sent = defaultdict(list)
    for temp in dic:
        annot = temp["annotations"]
        text = temp["text"]
        for section in annot:
            sec2sent[section["sectionLabel"]].append(text[section["begin"]: section["begin"] + section["length"]])
    return sec2sent


if __name__ == "__main__":
    json_path = "/Users/jongbeomkim/Documents/datasets/wikisection_dataset_json/wikisection_en_city_test.json"
    with open(json_path, mode="r") as f:
        dic = jsonable_encoder(json.load(f))
        sec2sent = _group_by_sections(dic)

    # permutations(sec2sent[sec1], 2)

    cnt = 0
    sections = list(sec2sent.keys())
    for sec1, sec2 in tqdm(list(permutations(sections, 2))):
        # for anchor, pos in permutations(sec2sent[sec1], 2):
        for anchor, pos in tqdm(list(combinations(sec2sent[sec1], 2))):
            for neg in sec2sent[sec2]:
                # neg
                cnt +=1
    cnt