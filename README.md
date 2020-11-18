# ATAE-LSTM
## what this project is
This is a realization of paper <a href="https://www.aclweb.org/anthology/D16-1058.pdf">Attention-based LSTM for Aspect-level Sentiment Classification</a>. It also contains some of my thinking and attempts for improvement.


This is my first attempt to build up an NLP project on my own in 2018, which contains some of the concepts that seem somewhat naive, e.g. the way to save memory by not importing embedding of words that never appear in dataset, and some of the technologies that seem old-fashion compared with <a href="https://github.com/huggingface/transformers">Transformers, BERT and their variants</a>. Nonetheless, it may be helpful to a beginner of NLP to get familiar with the concepts of word-embeddings, attention-mechanism, LSTM or GRU and how to build a simple NLP project using pytorch from scratch.


## how to run it
The entrance of training is *main_train.ipynb*, you can clone the whole project and run it through Jupyter Notebook in <a href="https://www.anaconda.com/">Anaconda</a>. The configuration of processing data, model structure and training details can be seen and modified in *config.py*. For visualization of the training process, you should run "python -m visdom.server" in command line.


The model's states during training will be saved in *checkpoints/* and you can return to one of these by *model.load(path)*.


*./data/glove/<a href="http://nlp.stanford.edu/data/glove.840B.300d.zip">glove.840B.300d</a>.txt* is needed for running, while you can also download and implement other pre-trained word vectors from <a href="https://nlp.stanford.edu/projects/glove/">Stanford-GloVe</a> if *embedding_root*, a hyperparameter declared in *config.py*, matches the real condition.


However, you can also create an empty file at this position and the programme will treat all the words in dataset as OOV words, \<UNKNOWN\>, which could have their own embedding once appear for *word_independence* times, another hyperparameter declared in *config.py*.



## what else I try
The quantities of the three labels, "positive", "negative" and "neutral", in dataset of sentiment classification in SemEval 2014 Task 4-2 is unbalanced, which is about 12/4/3. The training process seems to suffers from this as it can easily achieve 60% accuracy by simply classify all examples to be "positive", so I try to add a manual rescaling weight of different classes on the loss function. This change makes the validation accuracy more dynamic instead of keeping being the proportion of "positive" examples.


To help the model converge during training, I try to freeze or free (unfreeze) some of the parameters in a predetermined order as the layers neer to the output, e.g. the classifier, may vary quickly and randomly while the gradient directions of layers neer to the input, e.g. the aspect-attetion parameters, largely depend on these.


I also try to add Layer Normalization before activation functions in the aspect-attention mechanism to relieve vanishing gradient or exploding gradient. However, I have no idea whether it makes sense in this model as I didn't find a good hyperparameter setting on this structure, and someone said that <a href="https://zhuanlan.zhihu.com/p/74516930">Layer Normalization will not be influenced by batch_size but glove300d will.</a>.


What's more, I used to apply exponentially decreasing learning rate schedule such like *torch.optim.lr_scheduler.ReduceLROnPlateau* in training, which can break the state of *all positive* around 65-th epoch and achieve convergence after 300 epochs with initial learning rate of 1e-4, learning rate decay of 0.9 and minimum learning rate of 5e-6 (with no modification described above).


I look into the converged model by *mdoel.lin = None*, *%pdb on* and *score = model(sentence, terms)*, which call the *forward* method of model and block the method as the last linear layer is assigned to be *None* while *%pdb on* enables me to check the variables in the method stack. To my surprise, the converged model just calculated attention weights over input sequence to be equal, e.g. *[0.1, 0.1, ..., 0.1]*, and finally found that as the attention weights calculated by matrix multiplication is too small in values, e.g. 1e-3 ~ 1e-4, the softmax function can hardly distinguish between each tokens.
``` python
    >> nn.functional.softmax(t.Tensor([1, 2, 3]), dim=0)
    # tensor([0.0900, 0.2447, 0.6652])
    >> nn.functional.softmax(t.Tensor([0.1, 0.2, 0.3]), dim=0)
    # tensor([0.3006, 0.3322, 0.3672])
    >> nn.functional.softmax(t.Tensor([0.01, 0.02, 0.03]), dim=0)
    # tensor([0.3300, 0.3333, 0.3367])
```
This problem was solved by adding Layer Normalization before softmax.
```python
    >> layerNorm = nn.LayerNorm([3])
    >> nn.functional.softmax(layerNorm(t.Tensor([0.01, 0.02, 0.03])), dim=0)
    # tensor([0.0717, 0.2246, 0.7037], grad_fn=<SoftmaxBackward>)
```
After that the model reached 72% accuracy on validate set and can give right attention weights on tokens. The accuracy is still not as high as reported in the paper (even not as high as the baseline, LSTM) and I think it's because I only use 70% of the training-set and the left 30% is used as develop-set/validate-set (although in code it's named as test-set).


The attention mechanism seems brilliant when I first read the paper. However, it's actually generated through weighted sum of elements in two vectors, aspect-embedding and hidden-state, which mean the two vectors do not interact well compared to scorer functions in other attention mechanism. So I try to replace the original attention mechanism with simply *hidden_states projection_matrix aspect_embedding* where the project_matrix is the parameter of the model which is initialized as an identity matrix. This change can be turn on/off by the hyperparameter *use_myAttentionMechanism* in *config.py*.


## acknowledge
Thanks for the guidance from <a href="https://github.com/chenyuntc/pytorch-book">chenyuntc</a>.

