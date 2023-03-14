# Paper Summary
- [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
## Related Works
- Over the last few years, researchers have demonstrated the benefits of using word embeddings [11] [39] [42], which are trained on unlabeled corpora, to improve performance on a variety of tasks. These approaches, however, mainly transfer word-level information, whereas we aim to capture higher-level semantics.
- The closest line of work to ours involves pre-training a neural network using a language modeling objective and then fine-tuning it on a target task with supervision.
- Previous work proposed learning task specific architectures on top of transferred representations. Such an approach re-introduces a significant amount of task-specific customization and does not use transfer learning for these additional architectural components.
## Methodology
- Semi-supervised learning
    - we explore a semi-supervised approach for language understanding tasks using a combination of unsupervised pre-training and supervised fine-tuning.
    - ***Our goal is to learn a universal representation that transfers with little adaptation to a wide range of tasks. We assume access to a large corpus of unlabeled text and several datasets with manually annotated training examples (target tasks). Our setup does not require these target tasks to be in the same domain as the unlabeled corpus.***
    - Unsupervised pre-training is a special case of semi-supervised learning where the goal is to find a good initialization point instead of modifying the supervised learning objective. ***Subsequent research demonstrated that pre-training acts as a regularization scheme, enabling better generalization in deep neural networks.***
## Training
- We employ a two-stage training procedure. First, we use a language modeling objective on the unlabeled data to learn the initial parameters of a neural network model. Subsequently, we adapt these parameters to a target task using the corresponding supervised objective.
## Architecture
- Transformer
    - For our model architecture, we use the Transformer [62].
    - our choice of transformer networks allows us to capture longer-range linguistic structure.
- Finue-tune
    - Figure 1. Architecture for fune-tunning
        - <img src="https://i.imgur.com/MYuLqFT.png" width="800">
        - For some tasks, like text classification, we can directly fine-tune our model as described above. Certain other tasks, like question answering or textual entailment, have structured inputs such as ordered sentence pairs, or triplets of document, question, and answers. ***Since our pre-trained model was trained on contiguous sequences of text, we require some modifications to apply it to these tasks. We use a traversal-style approach, where we convert structured inputs into an ordered sequence that our pre-trained model can process. These input transformations allow us to avoid making extensive changes to the architecture across tasks. All transformations include adding randomly initialized start and end tokens (`<s>`, `<e>`).***
        - ***For entailment tasks, we concatenate the premise*** $p$ ***and hypothesis*** $h$ ***token sequences, with a delimiter token (`$`) in between.***
        - ***For similarity tasks, there is no inherent ordering of the two sentences being compared. To reflect this, we modify the input sequence to contain both possible sentence orderings (with a delimiter in between) and process each independently to produce two sequence representations*** $h^{m}_{l}$ ***which are added element-wise before being fed into the linear output layer.***
        - ***For question answering and commonsense reasoning, we are given a context document*** $z$***, a question*** $q$***, and a set of possible answers*** $\{a_{k}\}$***. We concatenate the document context and question with each possible answer, adding a delimiter token in between to get*** $[z; q; \$; a_{k}]$***. Each of these sequences are processed independently with our model and then normalized via a softmax layer to produce an output distribution over possible answers.***
## References
- [42] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- [62] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
