# References
    # https://nn.labml.ai/transformers/mlm/index.html

import sys
import torch

from bert.tokenize import prepare_bert_tokenizer

torch.set_printoptions(precision=2, edgeitems=12, linewidth=sys.maxsize, sci_mode=True)


class MaskedLanguageModeling(object):
    def __init__(
        self,
        vocab_size,
        mask_id,
        pad_id,
        no_mask_token_ids=[],
        select_prob=0.15,
        mask_prob=0.8,
        randomize_prob=0.1
    ):
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.no_mask_token_ids = no_mask_token_ids
        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.randomize_prob = randomize_prob

        self.no_mask_token_ids += [mask_id, pad_id]


    def __call__(self, x):
        cloned = x.clone()

        rand_tensor = torch.rand(x.shape, device=x.device)
        rand_tensor.masked_fill_(mask=torch.isin(x, torch.as_tensor(self.no_mask_token_ids)), value=1)

        # "Chooses 15% of the token positions at random for prediction."
        mask_mask = (rand_tensor < self.select_prob * self.mask_prob)
        randomize_mask = (rand_tensor >= self.select_prob * self.mask_prob) &\
            (rand_tensor < self.select_prob * (self.mask_prob + self.randomize_prob))

        # "If the $i$-th token is chosen, we replace the $i$-th token with
        # (1) the [MASK] token 80% of the time"
        x.masked_fill_(mask=mask_mask, value=mask_id)

        # "(2) a random token 10% of the time
        # (3) the unchanged $i$-th token 10% of the time."
        random_token_ids = torch.randint(high=self.vocab_size, size=torch.Size((randomize_mask.sum(),)))
        x[randomize_mask.nonzero(as_tuple=True)] = random_token_ids
        return x, cloned


if __name__ == "__main__":
    VOCAB_SIZE = 30_522
    vocab_path = "/Users/jongbeomkim/Desktop/workspace/transformer_based_models/bert/vocab_example.json"
    tokenizer = prepare_bert_tokenizer(vocab_path=vocab_path)
    mask_id = tokenizer.token_to_id("[MASK]")
    pad_id = tokenizer.token_to_id("[PAD]")
    mlm = MaskedLanguageModeling(vocab_size=VOCAB_SIZE, mask_id=mask_id, pad_id=pad_id)

    x1, x2 = mlm(token_ids)
    x1
    x2