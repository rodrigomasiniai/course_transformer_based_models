# Paper Summary
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
- Reference: [[Yonsei NLP Study] RoBERTa 발표](https://www.youtube.com/watch?v=_FUXSTK_Xqg&t=672s)
- Our modifications are simple, they include:
    - Training the model longer, with bigger batches, over more data;
    - Removing the next sentence prediction objective;
    - Training on longer sequences;
    - Dynamically changing the masking pattern applied to the training data.
- The contributions of this paper are: (1) We present a set of important BERT design choices and training strategies and introduce alternatives that lead to better downstream task performance;
## Training
- ***Past work in Neural Machine Translation has shown that training with very large mini-batches can both improve optimization speed and end-task performance when the learning rate is increased appropriately (Ott et al., 2018)***. Recent work has shown that BERT is also amenable to large batch training (You et al., 2019). Devlin et al. (2019) originally trained BERT-BASE for 1M steps with a batch size of 256 sequences. This is equivalent in computational cost, via gradient accumulation, to training for 125K steps with a batch size of 2K sequences, or for 31K steps with a batch size of 8K. ***We observe that training with large batches improves perplexity for the masked language modeling objective, as well as end-task accuracy. Large batches are also easier to parallelize via distributed data parallel training***, and in later experiments we train with batches of 8K sequences.
### Datasets
- We also collect a large new dataset (CC-NEWS) of comparable size to other privately used datasets, to better control for training set size effects.
- We confirm that using more data for pretraining further improves performance on downstream tasks.
### Masked Language Modeling
- Our training improvements show that masked language model pretraining, under the right design choices, is competitive with all other recently published methods. ***In the original implementation, random masking and replacement is performed once in the beginning and saved for the duration of training, although in practice, data is duplicated so the mask is not always the same for every training sentence.***
- We pretrain with sequences of at most $T = 512$ tokens.
- Unlike Devlin et al. (2019), we do not randomly inject short sequences, and we do not train with a reduced sequence length for the first 90% of updates. We train only with full-length sequences.
- Dynamic Masking
    - BERT relies on randomly masking and predicting tokens. ***The original BERT implementation performed masking once during data preprocessing, resulting in a single static mask. To avoid using the same mask for each training instance in every epoch, training data was duplicated 10 times so that each sequence is masked in 10 different ways over the 40 epochs of training. Thus, each training sequence was seen with the same mask four times during training.***
    - We compare this strategy with dynamic masking where we generate the masking pattern every time we feed a sequence to the model. This becomes crucial when pretraining for more steps or with larger datasets. ***We find that our reimplementation with static masking performs similar to the original BERT model, and dynamic masking is comparable or slightly better than static masking. Given these results and the additional efficiency benefits of dynamic masking, we use dynamic masking in the remainder of the experiments.***
### Next Sentence Prediction
- We compare several alternative training formats:
    - SEGMENT-PAIR + NSP: This follows the original input format used in BERT (Devlin et al., 2019), with the NSP loss. Each input has a pair of segments, which can each contain multiple natural sentences, but the total combined length must be less than 512 tokens.
    - SENTENCE-PAIR + NSP: Each input contains a pair of natural sentences, either sampled from a contiguous portion of one document or from separate documents. Since these inputs are significantly shorter than 512 tokens, we increase the batch size so that the total number of tokens remains similar to SEGMENT-PAIR + NSP. We retain the NSP loss.
    - FULL-SENTENCES: Each input is packed with full sentences sampled contiguously from one or more documents, such that the total length is at most 512 tokens. Inputs may cross document boundaries. When we reach the end of one document, we begin sampling sentences from the next document and add an extra separator token between documents. We remove the NSP loss.
    - DOC-SENTENCES: Inputs are constructed similarly to FULL-SENTENCES, except that they may not cross document boundaries. Inputs sampled near the end of a document may be shorter than 512 tokens, so we dynamically increase the batch size in these cases to achieve a similar number of total tokens as FULL-SENTENCES. We remove the NSP loss.
- We first compare the original SEGMENT-PAIR input format from Devlin et al. (2019) to the SENTENCE-PAIR format; both formats retain the NSP loss, but the latter uses single sentences. ***We find that using individual sentences hurts performance on downstream tasks, which we hypothesize is because the model is not able to learn long-range dependencies.***
- We next compare training without the NSP loss and training with blocks of text from a single document (DOC-SENTENCES). We find that this setting outperforms the originally published BERT-BASE results and that ***removing the NSP loss matches or slightly improves downstream task performance, in contrast to Devlin et al. (2019).*** It is possible that the original BERT implementation may only have removed the loss term while still retaining the SEGMENT-PAIR input format. Finally ***we find that restricting sequences to come from a single document (DOC-SENTENCES) performs slightly better than packing sequences from multiple documents (FULL-SENTENCES). However, because the DOC-SENTENCES format results in variable batch sizes, we use FULL-SENTENCES in the remainder of our experiments for easier comparison with related work.***
### Fine-tuning
- Our finetuning procedure follows thes original BERT paper.
## Evaluate
### Datasets
- SQuAD
    - We evaluate on two versions of SQuAD: V1.1 and V2.0 (Rajpurkar et al., 2016, 2018). In V1.1 the context always contains an answer, whereas in V2.0 some questions are not answered in the provided context, making the task more challenging. For SQuAD V1.1 we adopt the same span prediction method as BERT (Devlin et al., 2019). ***For SQuAD V2.0, we add an additional binary classifier to predict whether the question is answerable, which we train jointly by summing the classification and span loss terms. During evaluation, we only predict span indices on pairs that are classified as answerable.***
## Architecture
- We keep the model architecture fixed. Specifically, we begin by training BERT models with the same configuration as BERT-BASE

# Summary
- Training the model longer with more data (16GB -> 160GB)
    - Bookcorpus + English wikipedia: 16GB
    - CC-News: 76GB
    - Open Web Text: 38GB
    - Stories: 31GB
- Training with larger batches
    - Training with large mini-batches improve optimization speed and end-task performance.
    - BERT-base: 1M steps with batch size of 256 sequences
    - RoBERTa: 31k (500k?) steps with batch size of 8k sequences
- No next sentence prediction
    - Compare several alternative training formats (잘 이해가...)
        - Segment-pair + NSP: 각 Segment가 반드시 하나의 문장일 필요는 없음 (BERT)
        - Sentence-pair + NSP: 각 Segment가 반드시 하나의 문장임
        - Full Sentence: Document A의 마지막 문장과 Document B의 첫 번째 문장으로 구성될 수 있음
        - Doc Sentence: 반드시 하나의 Document에서 문장들이 추출되어야 함 (가장 우수한 성능)
- Dynamic, not static masking
- Text encoding
    - BERT: WordPiece, character-level BPE vocabulary of size 30k
    - RoBERTa: Subword vocabulary of a modest size (50k)
