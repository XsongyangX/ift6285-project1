#!/bin/bash

# Executes the entire machine learning pipeline
if [ $# -lt 4 ] || [ $# -gt 6 ]; then
    echo "Usage: ./learn_all.sh corpus_directory test_directory vectorize_kind model_kind [--distribute] [--new-vectorizer]"
    echo "ex. ./learn_all.sh ./data/excerpt ./data/test tfidf logistic_C20 --distribute"
    exit 1
fi
corpus=$1
test_set=$2

vectorizer=$3
model=$4

labels=(gender age zodiac)

if [ "$5" == "--new-vectorizer" ] || [ "$6" == "--new-vectorizer" ]; then
    python vectorizer.py $corpus data/vectorizers/$vectorizer.vec
fi

for label in "${labels[@]}"; do
    if [ $# -eq 4 ]; then
        ./learn.sh $corpus $test_set $vectorizer $model $label
    elif [ "$5" == "--distribute" ] || [ "$6" == "--distribute" ]; then
        pkscreen -S learn_$label ssh ens -J arcade \
            "cd ift6285/project1; ./learn.sh $corpus $test_set $vectorizer $model $label"
    fi
done
