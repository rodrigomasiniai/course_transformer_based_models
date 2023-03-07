# Byte Pair Encoding (BPE)
- Reference: https://huggingface.co/docs/transformers/main/tokenizer_summary
- Radford et al. (2019) introduce a clever implementation of BPE that uses bytes instead of unicode characters as the base subword units. Using bytes makes it possible to learn a subword vocabulary of a modest size (50K units) that can still encode any input text without introducing any "unknown" tokens.
- The original BERT implementation (Devlin et al., 2019) uses a character-level BPE vocabulary of size 30K, which is learned after preprocessing the input with heuristic tokenization rules. Following Radford et al. (2019), we instead consider training BERT with a larger byte-level BPE vocabulary containing 50K subword units, without any additional preprocessing or tokenization of the input. This adds approximately 15M and 20M additional parameters for BERT-BASE and BERT-LARGE, respectively.
- ***The vocabulary size, i.e. the base vocabulary size + the number of merges, is a hyperparameter to choose. For instance GPT has a vocabulary size of 40,478 since they have 478 base characters and chose to stop training after 40,000 merges.***
- With some additional rules to deal with punctuation, the GPT2's tokenizer can tokenize every text without the need for the "\<unk\>" symbol. GPT-2 has a vocabulary size of 50,257, which corresponds to the 256 bytes base tokens, a special end-of-text token and the symbols learned with 50,000 merges.
## Byte-level Byte Pair Encoding (BBPE)
- [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/pdf/1909.03341.pdf)

# UTF-8
- Reference: https://en.m.wikipedia.org/wiki/UTF-8
- UTF-8 is capable of encoding all 1,112,064[a] valid character code points in Unicode using one to four one-byte (8-bit) code units.
- The "x" characters are replaced by the bits of the code point
    - Code point ↔ UTF-8 conversion
        | First code point | Last code point | Byte 1 | Byte 2 | Byte 3 | Byte 4 | Code points |
        | - | - | - | - | - | - | - |
        | U+0000 | U+007F | 0xxxxxxx | | | | 128 |
        | U+0080 | U+07FF | 110xxxxx | 10xxxxxx | | |1,920 |
        | U+0800 | U+FFFF | 1110xxxx | 10xxxxxx | 10xxxxxx | | 61,440 |
        | U+10000 | U+10FFFF | 11110xxx | 10xxxxxx | 10xxxxxx | 10xxxxxx | 1,048,576 |

- 2진수 접두어: 0b, 8진수 접두어: 0o, 16진수 접두어: 0x