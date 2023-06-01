import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import json
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer
# from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer
from bert.wordpiece_implementation import collect_corpus, tokenize


tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
# tokenizer = BertWordPieceTokenizer("/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert-base-uncased-vocab.txt", lowercase=True)



def collect_corpus(corpus_dir):
    corpus_dir = Path(corpus_dir)

    corpus = list()
    for doc_path in tqdm(list(corpus_dir.glob("**/*.txt"))):
        # doc = ""
        # with open(doc_path, mode="r", encoding="utf-8") as f:
        #     for line in f:
        #         if line == "\n":
        #             continue
        #         line = line.replace("\n", "").replace("\t", "")
        #         doc += "\n" + line
        doc = [
            line.replace("\n", "").replace("\t", "")
            for line
            in open(doc_path, mode="r", encoding="utf-8")
            if line not in ["\n"]
        ]
        corpus.append(doc)
    return corpus


class BERTDataset(Dataset):
    def __init__(self, vocab_path, corpus_dir, seq_len):
        vocab_path = vocab_path
        corpus_dir = corpus_dir
        seq_len = seq_len

        vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab.json"
        with open(vocab_path, mode="r") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
        pad_id = vocab["[PAD]"]
        unk_id = vocab["[UNK]"]
        cls_id = vocab["[CLS]"]
        sep_id = vocab["[SEP]"]
        mask_id = vocab["[MASK]"]

        corpus_dir = "/Users/jongbeomkim/Documents/datasets/bookcorpus_subset"
        corpus = collect_corpus(corpus_dir=corpus_dir)
        for doc in corpus:
            tot_len = 0
            temp = list()
            for sent in doc:
                enc = tokenizer.encode(sent)
                ids = enc.ids

                if tot_len + len(ids) <= 512:
                    temp.extend(ids)
                    tot_len += len(ids)
            temp

# from tokenizers import Tokenizer
# from tokenizers.models import WordPiece
# from tokenizers.trainers import WordPieceTrainer

# tokenizer = Tokenizer(model=WordPiece())
# trainer = WordPieceTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])