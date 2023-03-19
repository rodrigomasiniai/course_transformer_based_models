# https://velog.io/@goggling/%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C%EC%99%80-UTF-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0
# https://velog.io/@zionhann/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C-%EB%AC%B8%EC%9E%90-%EB%B3%80%ED%99%98%ED%95%98%EA%B8%B0
# https://www.compart.com/en/unicode/U+7FD2
# https://konghana01.tistory.com/65

from transformers import GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer

pretrained_weights = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_weights)
text = "안녕하세요"
tokenizer.encode(text)
# dic = {idx: tokenizer.convert_ids_to_tokens([idx])[0] for idx in range(2 ** 8)}

tokenizer.convert_tokens_to_ids(tokenizer.convert_ids_to_tokens([256])[0])
tokenizer.convert_tokens_to_ids("녕")
tokenizer.convert_ids_to_tokens([50256])

tokenizer.convert_tokens_to_ids(["ㅁㄴㅇㄹ"])
tokenizer.tokenize("안녕하세요 저는 누구라고요?")

tokenizer.pad_token = tokenizer.eos_token

tokenizer.pad_token
tokenizer.eos_token
tokenizer.vocab_size


def tokenize(char):
    bytes = char.encode("utf-8")
    bytes
    hexes = bytes.hex()
    hexes
    tokenized = [chr(int(f"""0x{hexes[i: i + 2]}""", base=16)) for i in range(len(hexes))[:: 2]]
    return tokenized
tokenizer.encode(char)
[int(f"""0x{hexes[i: i + 2]}""", base=16) for i in range(len(hexes))[:: 2]]
[chr(int(f"""0x{hexes[i: i + 2]}""", base=16)) for i in range(len(hexes))[:: 2]]
"Ĵ"

chr(146)


char = "習"

tokenize(char)
tokenizer.tokenize(char)


[f"""0x{hexes[i: i + 2]}""" for i in range(len(hexes))[:: 2]]

tokenizer.tokenize(char)
tokenizer.convert_tokens_to_ids(tokenizer.tokenize(char))

[char.encode("utf-8").hex()


str(char.encode("utf-8")[1: 2])
hexes = str(char.encode("utf-8")[1: 2]).replace("b'\\", "0")
hexes
chr(int(hexes, base=16))
char.encode("utf-8").decode("utf-8")

tokenizer.tokenize(char)
chr(int("0xea", base=16)), chr(int("0xb3", base=16)), chr(int("0xa1", base=16))




[bin(i) for i in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(char))]
ord(char)
bin(ord(char))

ord("a")
hex(ord("a"))
hex(5)

chr(int("0x61", base=16))
chr(int("0xb3", base=16))

"a".encode("unicode_escape").hex()


ord(char.encode("utf-8")[1: 2])

"³".encode("unicode_escape")
"³".encode("unicode_escape").decode("utf-8")


bin(ord(char))
len(bin(ord(char)))
for i in range(4):
    bin(ord(char))[2 + 4 * i: 2 + 4 * (i + 1)]
int("0b11111010", base=2)
bin(166)


int(
    f"""0b{bin(ord(char))[2: 10]}""", base=2
)


int(bin(ord(char))[: 2 + 4], base=2)
bin(ord(char))[: 2 + 4]
char.encode("utf-8")

{
    "A": 10,
    "C": 12,
    "F": 15,
}

# "U+ACF0" -> 10 12 15 0
bin(10)
bin(12)