# https://medium.com/@pierre_guillou/byte-level-bpe-an-universal-tokenizer-but-aff932332ffe

# Byte Level BPE (BBPE) tokenizers from Transformers and Tokenizers (Hugging Face libraries)

# 1. Get the pre-trained GPT2 Tokenizer (pre-training with an English corpus)
from transformers import GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

pretrained_weights = "gpt2"
tokenizer_en = GPT2TokenizerFast.from_pretrained(pretrained_weights)
tokenizer_en.pad_token = tokenizer_en.eos_token

# 2. Train a Byte Level BPE (BBPE) tokenizer on the Portuguese Wikipedia

# Get GPT2 tokenizer_en vocab size
ByteLevelBPE_tokenizer_pt_vocab_size = tokenizer_en.vocab_size
ByteLevelBPE_tokenizer_pt_vocab_size

# ByteLevelBPETokenizer Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model

ByteLevelBPE_tokenizer_pt = ByteLevelBPETokenizer()


# Get list of paths to corpus files
path_data = Path("/Users/jongbeomkim/Desktop/workspace/transformer_based_models/gpt2")
# paths = [str(path_data/"corpus.txt")]
paths = [str(path_data/"empty.txt")]

# Customize training with <|endoftext|> special GPT2 token
ByteLevelBPE_tokenizer_pt.train(
    files=paths, 
    vocab_size=ByteLevelBPE_tokenizer_pt_vocab_size, 
    min_frequency=2, 
    special_tokens=["<|endoftext|>"]
)

# Get sequence length max of 1024
ByteLevelBPE_tokenizer_pt.enable_truncation(max_length=1024)

# save tokenizer
ByteLevelBPE_tokenizer_pt_rep = "ByteLevelBPE_tokenizer_pt"
path_to_ByteLevelBPE_tokenizer_pt_rep = path_data/ByteLevelBPE_tokenizer_pt_rep
if not (path_to_ByteLevelBPE_tokenizer_pt_rep).exists():
    path_to_ByteLevelBPE_tokenizer_pt_rep.mkdir(exist_ok=True, parents=True)
ByteLevelBPE_tokenizer_pt.save_model(str(path_to_ByteLevelBPE_tokenizer_pt_rep))

# 3. Import the tokenizer config files in Portuguese into the pre-trained GPT2 Tokenizer

# Get the path to ByteLevelBPE_tokenizer_pt config files
ByteLevelBPE_tokenizer_pt_rep = "ByteLevelBPE_tokenizer_pt"
path_to_ByteLevelBPE_tokenizer_pt_rep = path_data/ByteLevelBPE_tokenizer_pt_rep

# import the pre-trained GPT2TokenizerFast tokenizer with the tokenizer_pt config files
tokenizer_pt = GPT2TokenizerFast.from_pretrained(
    str(path_to_ByteLevelBPE_tokenizer_pt_rep), 
    pad_token="<|endoftext|>"
)

# Get sequence length max of 1024
tokenizer_pt.model_max_length = 1024


tokenizer_en.encode("곰")
tokenizer_en.decode(tokenizer_en.encode("곰"))