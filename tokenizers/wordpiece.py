# References
    # https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt
    # https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/vocab.py

# "Google never open-sourced its implementation of the training algorithm of WordPiece, so what follows
# is the best guess based on the published literature."

from transformers import AutoTokenizer
from collections import defaultdict
from tqdm.auto import tqdm
import json
import re

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")

# "Since it identifies subwords by adding a prefix (like `"##"` for BERT), each word is initially split
# by adding that prefix to all the characters inside the word. For instance, `'word'` gets split like; `'w ##o ##r ##d'`
# Thus, the initial alphabet contains all the characters present at the beginning of a word
# and the characters present inside a word preceded by the WordPiece prefix."


def pretokenize(text):
    pretokens = list()
    for i in re.split(pattern=r"[ ]+", string=text):
        for j in re.split(pattern=r"""([ !"#$%&'()*+,-./:;<=>?@\[\\\]^_`{\|}~]+)""", string=i):
            if j:
                pretokens.append(j)
    return pretokens


def get_pretoken_frequencies(corpus):
    freqs = defaultdict(int)
    for text in tqdm(corpus):
        pretokens = pretokenize(text)
        for pretoken, _ in pretokens:
            freqs[pretoken] += 1
    return freqs


def get_character_level_vocabulary(pretokens):
    vocab = list()
    for pretoken in pretokens:
        if pretoken[0] not in vocab:
            vocab.append(pretoken[0])
        for letter in pretoken[1:]:
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

    pair_scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return pair_scores


def build_vocab(freqs, splits, vocab_size):
    vocab = get_character_level_vocabulary(pretokens=freqs.keys())
    if len(vocab) >= vocab_size:
        return vocab

    with tqdm(total=vocab_size - len(vocab)) as pbar:
        while len(vocab) < vocab_size:
            pair_scores = _compute_pair_scores(freqs=freqs, splits=splits)
            best_pair, max_score = "", None
            for pair, score in pair_scores.items():
                if max_score is None or score > max_score:
                    best_pair = pair
                    max_score = score
            splits = _merge_pair(*best_pair, splits)
            new_token = (
                best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[0] + best_pair[1]
            )
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


def tokens_to_string(tokens):
    text = ""
    for token in tokens:
        if token[: 2] == "##":
            text += token[2:]
        else:
            text += " "
            text += token
    text = text[1:]
    text = re.sub(pattern=r"\[CLS\]|\[SEP\]", repl="", string=text)
    return text


if __name__ == "__main__":
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    freqs = get_pretoken_frequencies(corpus)
    splits = split_pretokens(pretokens=freqs.keys())
    vocab = build_vocab(freqs=freqs, splits=splits, vocab_size=1200)

    json_path = "vocab.json"
    with open(json_path, mode="w") as f:
        json.dump(vocab, f)
