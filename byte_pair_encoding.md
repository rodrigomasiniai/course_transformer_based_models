# Byte Pair Encoding (BPE)
- Reference: https://huggingface.co/docs/transformers/main/tokenizer_summary
- Radford et al. (2019) introduce a clever implementation of BPE that uses bytes instead of unicode characters as the base subword units. Using bytes makes it possible to learn a subword vocabulary of a modest size (50K units) that can still encode any input text without introducing any "unknown" tokens.
- The original BERT implementation (Devlin et al., 2019) uses a character-level BPE vocabulary of size 30K, which is learned after preprocessing the input with heuristic tokenization rules. Following Radford et al. (2019), we instead consider training BERT with a larger byte-level BPE vocabulary containing 50K subword units, without any additional preprocessing or tokenization of the input. This adds approximately 15M and 20M additional parameters for BERT-BASE and BERT-LARGE, respectively.
- ***The vocabulary size, i.e. the base vocabulary size + the number of merges, is a hyperparameter to choose. For instance GPT has a vocabulary size of 40,478 since they have 478 base characters and chose to stop training after 40,000 merges.***
- With some additional rules to deal with punctuation, the GPT2"s tokenizer can tokenize every text without the need for the "\<unk\>" symbol. GPT-2 has a vocabulary size of 50,257, which corresponds to the 256 bytes base tokens, a special end-of-text token and the symbols learned with 50,000 merges.
## Byte-level Byte Pair Encoding (BBPE)
- [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/pdf/1909.03341.pdf)
- The first 256 tokens are:
    ```python
    {0: "!", 1: '"', 2: "#", 3: "$", 4: "%", 5: "&", 6: "'", 7: "(", 8: ")", 9: "*", 10: "+", 11: ",", 12: "-", 13: ".", 14: "/", 15: "0", 16: "1", 17: "2", 18: "3", 19: "4", 20: "5", 21: "6", 22: "7", 23: "8", 24: "9", 25: ":", 26: ";", 27: "<", 28: "=", 29: ">", 30: "?", 31: "@", 32: "A", 33: "B", 34: "C", 35: "D", 36: "E", 37: "F", 38: "G", 39: "H", 40: "I", 41: "J", 42: "K", 43: "L", 44: "M", 45: "N", 46: "O", 47: "P", 48: "Q", 49: "R", 50: "S", 51: "T", 52: "U", 53: "V", 54: "W", 55: "X", 56: "Y", 57: "Z", 58: "[", 59: "\\", 60: "]", 61: "^", 62: "_", 63: "`", 64: "a", 65: "b", 66: "c", 67: "d", 68: "e", 69: "f", 70: "g", 71: "h", 72: "i", 73: "j", 74: "k", 75: "l", 76: "m", 77: "n", 78: "o", 79: "p", 80: "q", 81: "r", 82: "s", 83: "t", 84: "u", 85: "v", 86: "w", 87: "x", 88: "y", 89: "z", 90: "{", 91: "|", 92: "}", 93: "~", 94: "¡", 95: "¢", 96: "£", 97: "¤", 98: "¥", 99: "¦", 100: "§", 101: "¨", 102: "©", 103: "ª", 104: "«", 105: "¬", 106: "®", 107: "¯", 108: "°", 109: "±", 110: "²", 111: "³", 112: "´", 113: "µ", 114: "¶", 115: "·", 116: "¸", 117: "¹", 118: "º", 119: "»", 120: "¼", 121: "½", 122: "¾", 123: "¿", 124: "À", 125: "Á", 126: "Â", 127: "Ã", 128: "Ä", 129: "Å", 130: "Æ", 131: "Ç", 132: "È", 133: "É", 134: "Ê", 135: "Ë", 136: "Ì", 137: "Í", 138: "Î", 139: "Ï", 140: "Ð", 141: "Ñ", 142: "Ò", 143: "Ó", 144: "Ô", 145: "Õ", 146: "Ö", 147: "×", 148: "Ø", 149: "Ù", 150: "Ú", 151: "Û", 152: "Ü", 153: "Ý", 154: "Þ", 155: "ß", 156: "à", 157: "á", 158: "â", 159: "ã", 160: "ä", 161: "å", 162: "æ", 163: "ç", 164: "è", 165: "é", 166: "ê", 167: "ë", 168: "ì", 169: "í", 170: "î", 171: "ï", 172: "ð", 173: "ñ", 174: "ò", 175: "ó", 176: "ô", 177: "õ", 178: "ö", 179: "÷", 180: "ø", 181: "ù", 182: "ú", 183: "û", 184: "ü", 185: "ý", 186: "þ", 187: "ÿ", 188: "Ā", 189: "ā", 190: "Ă", 191: "ă", 192: "Ą", 193: "ą", 194: "Ć", 195: "ć", 196: "Ĉ", 197: "ĉ", 198: "Ċ", 199: "ċ", 200: "Č", 201: "č", 202: "Ď", 203: "ď", 204: "Đ", 205: "đ", 206: "Ē", 207: "ē", 208: "Ĕ", 209: "ĕ", 210: "Ė", 211: "ė", 212: "Ę", 213: "ę", 214: "Ě", 215: "ě", 216: "Ĝ", 217: "ĝ", 218: "Ğ", 219: "ğ", 220: "Ġ", 221: "ġ", 222: "Ģ", 223: "ģ", 224: "Ĥ", 225: "ĥ", 226: "Ħ", 227: "ħ", 228: "Ĩ", 229: "ĩ", 230: "Ī", 231: "ī", 232: "Ĭ", 233: "ĭ", 234: "Į", 235: "į", 236: "İ", 237: "ı", 238: "Ĳ", 239: "ĳ", 240: "Ĵ", 241: "ĵ", 242: "Ķ", 243: "ķ", 244: "ĸ", 245: "Ĺ", 246: "ĺ", 247: "Ļ", 248: "ļ", 249: "Ľ", 250: "ľ", 251: "Ŀ", 252: "ŀ", 253: "Ł", 254: "ł", 255: "Ń"}
    ```

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