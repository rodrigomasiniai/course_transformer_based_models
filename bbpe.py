from tokenizers import ByteLevelBPETokenizer

bytebpe_tokenizer = ByteLevelBPETokenizer()


char = "안"
char.encode("utf-8")[1]
chr(char.encode("utf-8")[1])
"".join([chr(i) for i in char.encode("utf-8")])


ord(char)
bin(ord(char))
f"""0b{bin(ord(char))[2: 10]}"""

bin(ord(char)).encode("utf-8")


"\xea"
chr(234)

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