# References
    # https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/vocab.py

from transformers import AutoTokenizer
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import json

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")


def collect_corpus(corpus_dir, add_empty_string=False):
    corpus_dir = Path(corpus_dir)

    corpus = list()
    for corpus_path in tqdm(list(corpus_dir.glob("**/*.txt"))):
        with open(corpus_path, mode="r", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    continue
                line = line.replace("\n", "").replace("\t", "")

                corpus.append(line)
    if add_empty_string:
        corpus.append("")
    return corpus


def get_pretoken_frequencies(corpus):
    freqs = defaultdict(int)
    for text in tqdm(corpus):
        pretokens = TOKENIZER.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        for pretoken, _ in pretokens:
            freqs[pretoken] += 1
    return freqs


def get_character_level_vocabulary(pretokens):
    vocab = list()
    for word in pretokens:
        if word[0] not in vocab:
            vocab.append(word[0])
        for letter in word[1:]:
            letter
            if f"##{letter}" not in vocab:
                vocab.append(f"##{letter}")
    vocab.sort()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + vocab.copy()

    vocab = {char: i for i, char in enumerate(vocab)}
    return vocab


def split_pretokens(pretokens):
    splits = {
        pretoken: [char if id_ == 0 else f"##{char}" for id_, char in enumerate(pretoken)]
        for pretoken in pretokens
    }
    return splits


def _merge_pair(a, b, splits):
    for pretoken in splits:
        split = splits[pretoken]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[pretoken] = split
    return splits


def _compute_pair_scores(freqs, splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores


def build_vocab(freqs, splits, vocab_size):
    vocab = get_character_level_vocabulary(pretokens=freqs.keys())
    if len(vocab) >= vocab_size:
        return vocab

    with tqdm(total=vocab_size - len(vocab)) as pbar:
        while len(vocab) < vocab_size:
            scores = _compute_pair_scores(freqs=freqs, splits=splits)
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or score > max_score:
                    best_pair = pair
                    max_score = score
            splits = _merge_pair(*best_pair, splits)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            # vocab.append(new_token)
            vocab[new_token] = len(vocab)

            pbar.update(1)
    return vocab


def _encode_word(word, vocab):
    tokens = list()
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[: i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]
        tokens.append(word[: i])

        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"
    return tokens


def tokenize(text, vocab):
    pretokens = TOKENIZER._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    encoded_words = [_encode_word(word=pretoken, vocab=vocab) for pretoken, _ in pretokens]
    return sum(encoded_words, [])


if __name__ == "__main__":
    corpus = collect_corpus("/Users/jongbeomkim/Documents/datasets/bookcorpus_subset")

    freqs = get_pretoken_frequencies(corpus)
    splits = split_pretokens(pretokens=freqs.keys())
    vocab = build_vocab(freqs=freqs, splits=splits, vocab_size=1200)

    json_path = "vocab.json"
    # json_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab.json"
    with open(json_path, mode="w") as f:
        json.dump(vocab, f)
