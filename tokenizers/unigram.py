# References
    # https://huggingface.co/learn/nlp-course/chapter6/7?fw=pt

# Unigram algorithm is often used in SentencePiece, which is the tokenization algorithm used by models like AlBERT, T5, mBART, Big Bird, and XLNet. 
# Compared to BPE and WordPiece, Unigram works in the other direction: it starts from a big vocabulary and removes tokens from it until it reaches the desired vocabulary size.  
# At each step of the training, the Unigram algorithm computes a loss over the corpus given the current vocabulary. Then, for each symbol in the vocabulary, the algorithm computes how much the overall loss would increase if the symbol was removed, and looks for the symbols that would increase it the least. Those symbols have a lower effect on the overall loss over the corpus, so in a sense they are “less needed” and are the best candidates for removal. 
# This is all a very costly operation, so we don’t just remove the single symbol associated with the lowest loss increase, but the �p (�p being a hyperparameter you can control, usually 10 or 20) percent of the symbols associated with the lowest loss increase. This process is then repeated until the vocabulary has reached the desired size. 
# Note that we never remove the base characters, to make sure any word can be tokenized. 
# A Unigram model is a type of language model that considers each token to be independent of the tokens before it. It’s the simplest language model, in the sense that the probability of token X given the previous context is just the probability of token X. So, if we used a Unigram language model to generate text, we would always predict the most common token. 
# The probability of a given token is its frequency (the number of times we find it) in the original corpus, divided by the sum of all frequencies of all tokens in the vocabulary (to make sure the probabilities sum up to 1). 
# Now, to tokenize a given word, we look at all the possible segmentations into tokens and compute the probability of each according to the Unigram model. Since all tokens are considered independent, this probability is just the product of the probability of each token. For instance, the tokenization ["p", "u", "g"] of "pug" has the probability:�([‘‘�",‘‘�",‘‘�"])=�(‘‘�")×�(‘‘�")×�(‘‘�")=5210×36210×20210=0.000389P([‘‘p",‘‘u",‘‘g"])=P(‘‘p")×P(‘‘u")×P(‘‘g")=2105​×21036​×21020​=0.000389 
# Comparatively, the tokenization ["pu", "g"] has the probability:�([‘‘��",‘‘�"])=�(‘‘��")×�(‘‘�")=5210×20210=0.0022676P([‘‘pu",‘‘g"])=P(‘‘pu")×P(‘‘g")=2105​×21020​=0.0022676 
# so that one is way more likely. In general, tokenizations with the least tokens possible will have the highest probability (because of that division by 210 repeated for each token), which corresponds to what we want intuitively: to split a word into the least number of tokens possible. 
# The tokenization of a word with the Unigram model is then the tokenization with the highest probability. In the example of "pug", here are the probabilities we would get for each possible segmentation: 
