## Resources for NLP models + DP

1. [Visualizing transformer models](https://jalammar.github.io/illustrated-transformer/): this is the basic
architecture used by many SOTA NLP models
   - A useful resource for testing out different tasks for NLP with transformers: https://github.com/tensorflow/tensor2tensor
   - A useful article particularly about BERT/ELMo models: https://jalammar.github.io/illustrated-bert/

2. [Sentiment Analysis Notebook](https://github.com/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb): a good working notebook
that can serve as a skeleton for training various models
   - This training is very resource intensive, so we can probably select
a small subset of the data to use for training (our goal is not necessarily
to achieve highest accuracy, but to show tradeoff with privacy)
     - We should be able to train the models on colab/kaggle, but if necessary we can probably also use UofT GPUs
   - I think we can start with some of the models available from [huggingface](https://huggingface.co/transformers/main_classes/model.html), e.g.: RoBERTa, GPT2, T5, etc..
   - Need to look into adding DP on top of these models in an easy manner:
     - in TF, this is done using https://github.com/tensorflow/privacy
     - in pytorch, we can try using: https://github.com/pytorch/opacus
         - in particular, we can check out: https://github.com/Darktex/opacus/blob/master/examples/imdb.py
         - torchtext has builtin datasets for various tasks: https://pytorch.org/text/datasets.html
