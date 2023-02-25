# Paper Summary
- Our modifications are simple, they include:
    - Training the model longer, with bigger batches, over more data;
    - Removing the next sentence prediction objective;
    - Training on longer sequences;
    - Dynamically changing the masking pattern applied to the training data.
- The contributions of this paper are: (1) We present a set of important BERT design choices and training strategies and introduce alternatives that lead to better downstream task performance;
## Train
### Datasets
- We also collect a large new dataset (CC-NEWS) of comparable size to other privately used datasets, to better control for training set size effects.
- We confirm that using more data for pretraining further improves performance on downstream tasks.
### MLM
- Our training improvements show that masked language model pretraining, under the right design choices, is competitive with all other recently published methods. **In the original implementation, random masking and replacement is performed once in the beginning and saved for the duration of training, although in practice, data is duplicated so the mask is not always the same for every training sentence.**

# RoBERTa
- Reference: https://www.youtube.com/watch?v=_FUXSTK_Xqg&t=672s
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
