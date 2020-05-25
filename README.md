# Commit message generation with Transformer

This is a PyTorch implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 

> The official Tensorflow Implementation can be found in: [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py).

> To learn more about self-attention mechanism, you could read "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)".

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>


The project support training and translation with trained model now.

Note that this project is still a work in progress.


If there is any suggestion or error, feel free to fire an issue to let me know. :)


# Requirement
- python 3.4+
- pytorch 0.4.1+
- tqdm
- numpy


# Usage
### 0) Download the data.
Download and unzip the dataset into the folder.

### 1) Preprocess the data.
```bash
python preprocess.py -train_src pathtodata/train_sourcefile -train_tgt pathtodata/train_targetfile -valid_src pathtodata/valid_sourcefile -valid_tgt pathtodata/valid_targetfile -save_data pathtodata/vocab -max_len 400 -min_word_count 0 -share_vocab
```

### 2) Train the model
```bash
python train.py -data pathtodata/vocab -save_model exp/model/ -log exp/log/ -save_mode best  -proj_share_weight -embs_share_weight -label_smoothing -epoch 100
```
> If your source and target language share one common vocabulary, use the `-embs_share_weight` flag to enable the model to share source/target word embedding. 

### 3) Test the model
```bash
python translate.py -model exp/model/trained.chkpt -vocab pathtodata/vocab -src pathtodata/test_sourcefile -output exp/resultfile
```
### 4) Evaluate the result
```bash
python evaluate/evaluate.py pathto/candidate pathto/reference
perl evaluate/multi-bleu.perl pathto/reference < pathto/candidate
```
---
# Performance

---
# TODO

---
# Acknowledgement

