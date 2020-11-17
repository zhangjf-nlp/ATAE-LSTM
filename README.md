# ATAE-LSTM
realization of paper 《Attention-based LSTM for Aspect-level Sentiment Classification》


*./data/glove/<a href="http://nlp.stanford.edu/data/glove.840B.300d.zip">glove.840B.300d</a>.txt* is needed for running, while you can also download and implement other pre-trained word vectors from <a href="https://nlp.stanford.edu/projects/glove/">Stanford-GloVe</a> if *embedding_root*, a hyperparameter declared in *config.py*, matches the real condition.


However, you can also create an empty file at this position and the programme will treat all the words in dataset as OOV words, \<UNKNOWN\>, which could have their own embedding once appear for *word_independence* times, another hyperparameter declared in *config.py*.


This is my first attempt to build up an NLP project on my own in 2018, which contains some of the concepts that seem somewhat naive, e.g. the way to save memory by not importing embedding of words that never appear in dataset, and some of the technologies that seem old-fashion compared with <a href="https://github.com/huggingface/transformers">Transformers, BERT and their variants</a>. Nonetheless, it may be helpful to a beginner of NLP to get familiar with the concepts of word-embeddings, attention-mechanism, LSTM or GRU and how to build a simple NLP project using pytorch from scratch.


Thanks for the guidance of <a href="https://github.com/chenyuntc/pytorch-book">chenyuntc</a>.
