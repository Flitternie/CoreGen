# !/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 Reference Prediction" >&2
    exit 1
fi

REF=$1
PRED=$2

perl ./evaluation/multi-bleu.perl -lc ${REF} < ${PRED}
python ./evaluation/evaluate.py ${REF} ${PRED}