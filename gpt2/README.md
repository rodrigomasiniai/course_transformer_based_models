# Paper Summary
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
## Related Works
- The current best performing systems on language tasks utilize a combination of pre-training and supervised fine-tuning. This approach has a long history with a trend towards more flexible forms of transfer. First, word vectors were learned and used as inputs to task-specific architec-tures, then the contextual representations of recurrent networks were transferred [1], and recent work suggests that task-specific architectures are no longer necessary and transferring many self-attention blocks is sufficient [2] [3].
- These methods still require supervised training in order to perform a task. When only minimal or no supervised data is available, another line of work has demonstrated the promise of language models to perform specific tasks, such as commonsense reasoning and sentiment analysis [2].
## Methodology
- Unsupervised multitask & zero-shot learning
    - We would like to move towards more general systems which can perform many tasks – eventually without the need to manually create and label a training dataset for each one.
    - Our suspicion is that the prevalence of single task training on single domain datasets is a major contributor to the lack of generalization observed in current systems.
    - We demonstrate language models can perform down-stream tasks in a zero-shot setting – without any parameter or architecture modification.
    - Learning to perform a single task can be expressed in a probabilistic framework as estimating a conditional distribution $p(output \mid input)$. ***Since a general system should be able to perform many different tasks, even for the same input, it should condition not only on the input but also on the task to be performed. That is, it should model*** $p(output \mid input, task)$***.***
    - Language provides a flexible way to specify tasks, inputs, and outputs all as a sequence of symbols. For example, a translation training example can be written as the sequence '(translate to french, english text, french text)'. Likewise, a reading comprehension training example can be written as '(answer the question, document, question, answer)'.
    - Our speculation is that a language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them, regardless of their method of procurement. If a language model is able to do this it will be, in effect, performing unsupervised multitask learning. We test whether this is the case by analyzing the performance of language models in a zero-shot setting on a wide variety of tasks.
- Tokenization
    - ***Current large scale LMs include pre-processing steps such as lower-casing, tokenization, and out-of-vocabulary tokens which restrict the space of model-able strings.*** While processing Unicode strings as a sequence of UTF-8 bytes elegantly fulfills this requirement as exemplified in work such as [5], ***current byte-level LMs are not competitive with word-level LMs on large scale datasets. We observed a similar performance gap in our own attempts to train standard byte-level LMs on WebText.***
    - ***A byte-level version of BPE only requires a base vocabulary of size 256. However, directly applying BPE to the byte sequence results in suboptimal merges due to BPE using a greedy frequency based heuristic for building the token vocabulary.*** This results in a suboptimal allocation of limited vocabulary slots and model capacity. To avoid this, we prevent BPE from merging across character categories for any byte sequence. We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.
    - ***Since our approach can assign a probability to any Unicode string, this allows us to evaluate our LMs on any dataset regardless of pre-processing, tokenization, or vocab size.***
- Language modeling
    - Language modeling is usually framed as unsupervised distribution estimation from a set of examples $(x_{1}, x_{2}, ..., x_{n})$ each composed of variable length sequences of symbols $(s_{1}, s_{2}, ..., s_{n})$. Since language has a natural sequential ordering, it is common to factorize the joint probabilities over symbols as the product of conditional probabilities:
    $$p(x) = \prod_{i = 1}^{n}p(s_{i}|s_{1}, s_{2}, ..., s_{n - i})$$
## Architecture
- We use a Transformer [4] based architecture for our LMs. The model largely follows the details of the OpenAI GPT model [2] with a few modifications. ***Layer normalization [6] was moved to the input of each sub-block, similar to a pre-activation residual network [7] and an additional layer normalization was added after the final self-attention block. A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of $\frac{1}{\sqrt{N}}$ where $N$ is the number of residual layers. The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.***
- Table 2. Model variants
    - <img src="https://miro.medium.com/v2/resize:fit:618/format:webp/1*xfuKZBGFVfryzfi7smKTQg.png" width="200">
    - The smallest model is equivalent to the original GPT, and the second smallest equivalent to the largest model from BERT [3]. ***Our largest model, which we call GPT-2***, has over an order of magnitude more parameters than GPT.
## Training
### Datasets
- WebText
    - We created a new web scrape which emphasizes document quality. To do this we only scraped web pages which have been curated/filtered by humans.
    - Contains slightly over 8 million documents for a total of 40 GB of text. We removed all Wikipedia documents from WebText since it is a common data source for other datasets and could complicate analysis due to over lapping training data with test evaluation tasks.
- Table 3
    
## Experiments
## References
- [1] [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
- [2] [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [3] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [4] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [5] [Multilingual Language Processing From Bytes](https://arxiv.org/pdf/1512.00103.pdf)
- [6] [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
- [7] [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)

- Official repository: https://github.com/openai/gpt-2