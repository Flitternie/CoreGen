# !/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0" >&2
    exit 1
fi

DATAPATH=./data
SAVEPATH=./exp

LAYER=$1
HEAD=$2

mkdir ${SAVEPATH}
mkdir ${SAVEPATH}/vocab/
mkdir ${SAVEPATH}/log/
mkdir ${SAVEPATH}/pretrain/
mkdir ${SAVEPATH}/finetune/

python preprocess.py -train_src ${DATAPATH}/cleaned.train.diff -train_tgt ${DATAPATH}/cleaned.train.msg -valid_src ${DATAPATH}/cleaned.valid.diff -valid_tgt ${DATAPATH}/cleaned.valid.msg -save_data ${SAVEPATH}/vocab/vocab -max_len 300 -min_word_count 0 -share_vocab
python pretrain.py -train_src ./data/cleaned.train.diff -valid_src ${DATAPATH}/cleaned.valid.diff -vocab ${SAVEPATH}/vocab/vocab -save_data ${SAVEPATH}/vocab/pretrain_vocab -mask_rate 0.5 -max_len 300 -min_word_count 0

python train.py -data ${SAVEPATH}/vocab/pretrain_vocab -save_model ${SAVEPATH}/pretrain/pretrain_${LAYER}layer_${HEAD}head_0.5mask -log ${SAVEPATH}/log/pretrain_${LAYER}layer_${HEAD}head_0.5mask -save_mode best -proj_share_weight -embs_share_weight -label_smoothing -epoch 2 -batch_size 16 -n_head ${HEAD} -n_layers ${LAYER}
BEST_PRETRAIN=$(ls "$SAVEPATH"/pretrain/ | grep pretrain_"${LAYER}"layer_"${HEAD}"head* | tail -n1)

python train.py -data ${SAVEPATH}/vocab/vocab -model ${SAVEPATH}/pretrain/${BEST_PRETRAIN} -save_model ${SAVEPATH}/finetune/finetune_${LAYER}layer_${HEAD}head_0.5mask -log ${SAVEPATH}/log/finetune_${LAYER}layer_${HEAD}head_0.5mask -save_mode best -proj_share_weight -embs_share_weight -label_smoothing -epoch 2 -batch_size 32 -n_head ${HEAD} -n_layers ${LAYER}
BEST_MODEL=$(ls "$SAVEPATH"/finetune/ | grep finetune_"${LAYER}"layer_"${HEAD}"head* | tail -n1)

mkdir ${SAVEPATH}/result/
python translate.py -model ./exp/finetune/${BEST_MODEL} -vocab ${SAVEPATH}/vocab/vocab -src ${DATAPATH}/cleaned.test.diff -output ${SAVEPATH}/result/${LAYER}layer_${HEAD}head_0.5mask.msg