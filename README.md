# Exploiting Structural Code Embedding for Augmented Code Commit Message Generation
This is the source code repository for the paper "Exploiting Structural Code Embedding for Augmented Code Commit Message Generation".

Note that this project is still a work in progress. If there is any suggestion or error, feel free to fire an issue to let me know.

# Requirement
- python 3.4+
- pytorch 0.4.1+
- tqdm
- numpy

# Usage
## 0) Download the data.
Download the data [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155079751_link_cuhk_edu_hk/EXsJ_2t1qtJHlFz9FEQe3swBx-Atm31Sg0cBbiDq6dW7ag?e=lUTeQQ) and unzip the dataset into the folder.

## 1) Preprocess the data.
```bash
python preprocess.py -train_src data/cleaned.train.diff -train_tgt data/cleaned.train.msg -valid_src data/cleaned.valid.diff -valid_tgt data/cleaned.valid.msg -save_data exp/vocab/vocab -max_len 300 -min_word_count 0 -share_vocab
```
```bash
python pretrain.py -train_src ./data/cleaned.train.diff -valid_src ./data/cleaned.valid.diff -vocab ./exp/vocab/vocab -save_data ./exp/vocab/pretrain_vocab -mask_rate 0.5 -max_len 300 -min_word_count 0
```

## 2) Train the model
### a) Self-supervised Code Embedding Exploitation
```bash
python train.py -data exp/vocab/pretrain_vocab -save_model exp/pretrain/pretrain_2layer_40epoch_6head_0.5maskrate -log exp/log/pretrain_2layer_40epoch_6head_0.5maskrate -save_mode best -save_thres 0.85 -proj_share_weight -embs_share_weight -label_smoothing -epoch 40 -batch_size 16 -n_head 6 -n_layers 2
```
> Adjust the ```save_thres``` parameter to define the model saving threshold

### b) Supervised Commit Message Generation
```bash
python train.py -data exp/vocab/vocab -model exp/pretrain/pretrain_2layer_40epoch_6head_0.5maskrate_accu_XXX.chkpt -save_model exp/finetune/finetune_2layer_100epoch_6head_0.5maskrate -log exp/log/finetune_2layer_100epoch_6head_0.5maskrate -save_mode best -save_thres 0.35 -proj_share_weight -embs_share_weight -label_smoothing -epoch 100 -batch_size 32 -n_head 6 -n_layers 2
```
> Adjust the ```save_thres``` parameter to define the model saving threshold

## 3) Inference
```bash
python translate.py -model ./exp/pretrain/finetune_2layer_100epoch_6head_0.5maskrate_accu_XXX.chkpt -vocab exp/vocab/vocab -src ./data/cleaned.test.diff -output exp/result/finetuned_2layer_0.5maskrate.msg
```

## 4) Result Evaluation
Switch to python 2.7 for the following executions:
```bash
python evaluate/evaluate.py pathto/candidate pathto/reference
perl evaluate/multi-bleu.perl pathto/reference < pathto/candidate
```
---
# Performance
- To be updated
---

