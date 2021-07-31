#!/bin/sh

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 LAYER_NUM HEAD_NUM (Example: ./run_all.sh 2 6)" >&2
    exit 1
fi

DATAPATH=./data
SAVEPATH=./exp

# hyper-parameter configuration
LAYER=$1
HEAD=$2
STAGE1_BATCH_SIZE=32
STAGE2_BATCH_SIZE=64
STAGE1_EPOCH=30
STAGE2_EPOCH=100
MASK_RATE=0.5
MAX_LENGTH=512
MIN_WORD_COUNT=0

# preparing directories
mkdir -p ${SAVEPATH}
mkdir -p ${SAVEPATH}/vocab/
mkdir -p ${SAVEPATH}/log/
mkdir -p ${SAVEPATH}/pretrain/
mkdir -p ${SAVEPATH}/finetune/

# preprocessing
python preprocess.py -train_src ${DATAPATH}/cleaned.train.diff -train_tgt ${DATAPATH}/cleaned.train.msg -valid_src ${DATAPATH}/cleaned.valid.diff -valid_tgt ${DATAPATH}/cleaned.valid.msg -save_data ${SAVEPATH}/vocab/vocab -max_len ${MAX_LENGTH} -min_word_count ${MIN_WORD_COUNT} -share_vocab
python pretrain.py -train_src ${DATAPATH}/cleaned.train.diff -valid_src ${DATAPATH}/cleaned.valid.diff -vocab ${SAVEPATH}/vocab/vocab -save_data ${SAVEPATH}/vocab/pretrain_vocab -mask_rate ${MASK_RATE} -max_len ${MAX_LENGTH} -min_word_count ${MIN_WORD_COUNT}  

# stage I training
python train.py -data ${SAVEPATH}/vocab/pretrain_vocab -save_model ${SAVEPATH}/pretrain/pretrain_${LAYER}layer_${HEAD}head_${MASK_RATE}mask -log ${SAVEPATH}/log/pretrain_${LAYER}layer_${HEAD}head_${MASK_RATE}mask -save_mode best -proj_share_weight -embs_share_weight -label_smoothing -epoch ${STAGE1_EPOCH} -batch_size ${STAGE1_BATCH_SIZE} -n_head ${HEAD} -n_layers ${LAYER}

# stage II training 
BEST_PRETRAIN=$(ls "$SAVEPATH"/pretrain/ | grep pretrain_"${LAYER}"layer_"${HEAD}"head* | tail -n1)
python train.py -data ${SAVEPATH}/vocab/vocab -model ${SAVEPATH}/pretrain/${BEST_PRETRAIN} -save_model ${SAVEPATH}/finetune/finetune_${LAYER}layer_${HEAD}head_${MASK_RATE}mask -log ${SAVEPATH}/log/finetune_${LAYER}layer_${HEAD}head_${MASK_RATE}mask -save_mode best -proj_share_weight -embs_share_weight -label_smoothing -epoch ${STAGE2_EPOCH} -batch_size ${STAGE2_BATCH_SIZE} -n_head ${HEAD} -n_layers ${LAYER}

# inference
mkdir -p ${SAVEPATH}/result/
BEST_MODEL=$(ls "$SAVEPATH"/finetune/ | grep finetune_"${LAYER}"layer_"${HEAD}"head* | tail -n1)
python translate.py -model ${SAVEPATH}/finetune/${BEST_MODEL} -vocab ${SAVEPATH}/vocab/vocab -src ${DATAPATH}/cleaned.test.diff -output ${SAVEPATH}/result/${LAYER}layer_${HEAD}head_${MASK_RATE}mask.msg