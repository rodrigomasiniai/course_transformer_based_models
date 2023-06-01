# References
    # https://huggingface.co/transformers/v3.1.0/_modules/transformers/tokenization_bert.html
    # https://blog.naver.com/PostView.naver?blogId=sooftware&logNo=222494375953&parentCategoryNo=&categoryNo=13&viewDate=&isShowPopularPosts=false&from=postView
    # https://velog.io/@nawnoes/Huggingface-tokenizers%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%9C-Wordpiece-Tokenizer-%EB%A7%8C%EB%93%A4%EA%B8%B0
    # https://cryptosalamander.tistory.com/139


from tokenizers import BertWordPieceTokenizer
# from transformers import PreTrainedTokenizerFast
# from transformers import AutoTokenizer
from transformers import BertTokenizer

vocab_path = "../data/wpm-vocab-all.txt"

tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)
from pathlib import Path
import json # import json module

VOCAB_SIZE = 3000


# def train_wordpiece_tokenizer(data_files, vocab_size, save_dir):
#     save_dir = Path(save_dir)

#     tokenizer = BertWordPieceTokenizer(
#         clean_text=True, # hether to clean the text before tokenization by removing any control characters
#             # and replacing all whitespaces by the classic one.
#         handle_chinese_chars=True, # Whether to tokenize Chinese characters.
#             # This should likely be deactivated for Japanese:
#         strip_accents=False, # Must be False if cased model / 액센트 제거
#         lowercase=False, # Whether to lowercase the input when tokenizing.
#         wordpieces_prefix="##" # The prefix for subwords.
#     )
#     tokenizer.train(
#         files=data_files,
#         limit_alphabet=0, # The maximum different characters to keep in the alphabet.
#         vocab_size=vocab_size
#     )

#     json_path = save_dir/f"tokenizer.json"
#     tokenizer.save(str(json_path))

#     tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(json_path))
#     tokenizer.save_pretrained(save_dir)

#     tokenizer = AutoTokenizer.from_pretrained(save_dir)
#     return tokenizer


def train_wordpiece_tokenizer(data_files, vocab_size, json_path):
    tokenizer = BertWordPieceTokenizer(
        clean_text=True, # hether to clean the text before tokenization by removing any control characters
            # and replacing all whitespaces by the classic one.
        handle_chinese_chars=True, # Whether to tokenize Chinese characters.
            # This should likely be deactivated for Japanese:
        strip_accents=False, # Must be False if cased model / 액센트 제거
        lowercase=False, # Whether to lowercase the input when tokenizing.
        wordpieces_prefix="##" # The prefix for subwords.
    )
    tokenizer.train(
        files=data_files,
        limit_alphabet=0, # The maximum different characters to keep in the alphabet.
        vocab_size=vocab_size
    )
    tokenizer.save(str(json_path))


def json_to_txt_for_vocab(json_path):
    txt_path = f"""{json_path.rsplit(".", 1)[0]}.txt"""
    with open(txt_path, mode="w",encoding="utf-8") as txt_file:
        with open(json_path) as json_file:
            json_data = json.load(json_file)
            for item in json_data["model"]["vocab"].keys():
                txt_file.write(item+"\n")
            txt_file.close()

data_files=["/Users/jongbeomkim/Documents/datasets/wikisection_dataset_json/wikisection_en_city_test.json"]
json_path="/Users/jongbeomkim/Downloads/wp.json"
tokenizer = train_wordpiece_tokenizer(data_files=data_files, vocab_size=VOCAB_SIZE, vocab_path=json_path)
json_to_txt_for_vocab(json_path)
tokenizer = BertTokenizer(vocab_file="/Users/jongbeomkim/Downloads/wp.txt", do_lower_case=False)

tokenizer.encode("문장", add_special_tokens=False)



tokenizer.tokenize("튜닙은 자연어처리 테크 스타트업이다.")