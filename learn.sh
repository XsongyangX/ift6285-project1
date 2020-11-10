#!/bin/bash

# Executes the learning pipeline for a given label
if [ $# -ne 5 ] && [ $# -ne 6 ]; then
    echo "Usage: ./learn.sh corpus test-set vectorizer_kind model_kind label [--new-vectorizer]"
    echo "ex. ./learn.sh ./data/excerpt ./data/test tfidf logistic gender --new-vectorizer"
fi

corpus=$1
test_set=$2
vectorizer=$3
model=$4
label=$5

if [ $# -eq 6 ]; then 
    python vectorizer.py $corpus data/vectorizers/$vectorizer.vec
fi

python train.py $corpus data/vectorizers/$vectorizer.vec \
    data/models/$label-$model-$vectorizer.model \
    --label $label

python predict.py $test_set data/models/$label-$model-$vectorizer.model \
    data/vectorizers/$vectorizer.vec data/predictions/$label-$model-$vectorizer.predictions \
    --label $label

mkdir -p data/performances
python evaluate.py data/predictions/$label-$model-$vectorizer.predictions \
    --plot data/performances/$label-$model-$vectorizer.png \
    > data/performances/$label-$model-$vectorizer.log
