from gensim.models.word2vec import Word2Vec
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

import pandas as pd
import sys, os
from pickle import dump, load
from typing import Union
import csv

from corpus.preprocessing import Preprocessor

# config
csv.field_size_limit(100000000)
Vectorizer = Union[CountVectorizer, TfidfVectorizer, HashingVectorizer, Word2Vec]

class Training:
    """Manages the training environment of a classifier
    """

    def __init__(self, model: ClassifierMixin,\
        vectorizer: Vectorizer,\
        training_data: Preprocessor,\
        vectorizer_from_disk:str=None):
        """
        Creates a training environment for the model
        """
        self.model = model
        self.vectorizer = vectorizer
        self.training_data = training_data

        if vectorizer_from_disk is not None:
            with open(vectorizer_from_disk, "rb") as disk:
                self.vectorizer : Vectorizer = load(disk)
        else:
            if type(self.vectorizer) is not Word2Vec:
                def get_documents():
                    for blog, _ in self.training_data.run_yield():
                        yield " ".join(blog)
                self.vectorizer.fit(get_documents())
            else:
                raise Exception("Word2vec must be loaded from disk")

    def train(self, target:str="gender"):
        """
        Training the model with respect to the given labels
        """
        def vectorize():
            for blog, _ in self.training_data.run_yield(no_dumping=False):                
                yield self.vectorizer.transform("".join(blog))
        
        def labels_from_temp():
            for labels in self.training_data.load_temp_labels():
                yield getattr(labels, target)
        
        self.model.fit([vectorize()], labels_from_temp())

def main():
    from sklearn.naive_bayes import MultinomialNB
    training = Training(MultinomialNB(), CountVectorizer(), Preprocessor("data/excerpt"))
    training.train()

if __name__ == "__main__":
    main()