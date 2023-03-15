- Official repository: https://github.com/openai/gpt-2

# Paper Summary
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- Unsupervised learning
    - We would like to move towards more general systems which can perform many tasks – eventually without the need to manually create and label a training dataset for each one.
    - Our suspicion is that the prevalence of single task training on single domain datasets is a major contributor to the lack of generalization observed in current systems.
## Tokenization
- ***A byte-level version of BPE only requires a base vocabulary of size 256. However, directly applying BPE to the byte sequence results in suboptimal merges due to BPE using a greedy frequency based heuristic for building the token vocabulary.*** This results in a suboptimal allocation of limited vocabulary slots and model capacity. To avoid this, we prevent BPE from merging across character categories for any byte sequence. We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.
## Related Works
- The current best performing systems on language tasks utilize a combination of pre-training and supervised fine-tuning. This approach has a long history with a trend towards more flexible forms of transfer. First, word vectors were learned and used as inputs to task-specific architec-tures, then the contextual representations of recurrent networks were transferred [1], and recent work suggests that task-specific architectures are no longer necessary and transferring many self-attention blocks is sufficient [2] [3].
- These methods still require supervised training in order to perform a task. When only minimal or no supervised data is available, another line of work has demonstrated the promise of language models to perform specific tasks, such as commonsense reasoning and sentiment analysis [2].

- Zero-shot learning
    - We demonstrate language models can perform down-stream tasks in a zero-shot setting – without any parameter or architecture modification. We achieve promising, competitive, and state of the art results depending on the task.
- Language modeling
    - At the core of our approach is language modeling. Language modeling is usually framed as unsupervised distribution estimation from a set of examples $(x_{1}, x_{2}, ..., x_{n})$ each composed of variable length sequences of symbols $(s_{1}, s_{2}, ..., s_{n})$. Since language has a natural sequential ordering, it is common to factorize the joint probabilities over symbols as the product of conditional probabilities:
    $$p(x) = \prod_{i = 1}^{n}p(s_{i}|s_{1}, s_{2}, ..., s_{n - i})$$
    - This approach allows for tractable sampling from and estimation of $p(x)$ as well as any conditionals of the form $p(s_{n − k}, s_{n - k + 1} ..., s_{n}|s_{1}, s_{2}, ..., s_{n − k − 1})$. In recent years, there have been significant improvements in the expressiveness of models that can compute these conditional probabilities, such as self-attention architectures like the Transformer [4].
## References
- [1] [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
- [2] [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [3] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [4] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
