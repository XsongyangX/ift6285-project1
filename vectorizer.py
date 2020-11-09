# Fit a vectorizer from the data set and saves it to disk
# This is where preprocessing is done
import argparse
import os
from pickle import dump
import pandas as pd
from pandas.core.frame import DataFrame

def get_args():
    parser = argparse.ArgumentParser(description="Trains a vectorizer and saves it to disk")

    parser.add_argument("corpus", help="Directory of the blog csv corpus")
    parser.add_argument("save", help="Save location for the vectorizer")

    return parser.parse_args()

def main():
    args = get_args()

    # Get corpus
    training_data : DataFrame = None
    for csv_file in os.listdir(args.corpus):
        with open(os.path.join(args.corpus, csv_file), encoding='utf-8') as file:
            data = pd.read_csv(file, names=['bloggerID', 'gender', 'age', 'zodiac', 'blog'])
            if training_data is None:
                training_data = data
            else:
                training_data = training_data.append(data)

    # EDIT HERE
    # Fit vectorizer, put preprocessing
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize import TweetTokenizer
    tweet = TweetTokenizer()
    vectorizer = TfidfVectorizer(tokenizer=tweet.tokenize, lowercase=False)

    vectorizer.fit(training_data['blog'])

    # Save vectorizer
    save_folder, _ = os.path.split(args.save)
    os.makedirs(save_folder, exist_ok=True)

    with open(args.save, "wb") as model:
        dump(vectorizer, model)
        
if __name__ == "__main__":
    main()