#!/bin/bash

# Executes the entire machine learning pipeline
if [ $# -ne 4 ] && [ $# -ne 5 ]; then
    echo "Usage: ./learn_all.sh corpus-directory test-directory vectorize-kind model-kind [--distribute]"
    echo "ex. ./learn_all.sh ./data/excerpt ./data/test tfidf logistic --distribute"
    exit 1
fi
corpus=$1
test_set=$2

vectorizer=$3
model=$4

labels=(gender age zodiac)

python vectorizer.py $corpus data/vectorizers/$vectorizer.vec

for label in "${labels[@]}"; do
    if [ $# -eq 4 ]; then
        ./learn.sh $corpus $test_set $vectorizer $model $label
    elif [ $# -eq 5 ]; then
        pkscreen -S learn_$label ssh ens -J arcade \
            "cd ift6285/project1; ./learn.sh $corpus $test_set $vectorizer $model $label"
    fi
done
