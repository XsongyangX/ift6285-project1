# Train using vectorizers saved on disk
import argparse
import os
from pickle import dump, load
import pandas as pd
from pandas.core.frame import DataFrame

def get_args():
    parser = argparse.ArgumentParser(description="Trains a model and saves it to disk")

    parser.add_argument("corpus", help="Directory of the blog csv corpus")
    parser.add_argument("vectorizer", help="Vectorizer file")
    parser.add_argument("save", help="Save location for the model")

    parser.add_argument("--label", help="Which label to train for, ex. 'gender' (default), 'age', 'zodiac'",\
        default="gender")

    return parser.parse_args()

def main():
    args = get_args()

    # Load vectorizer
    vectorizer = load(open(args.vectorizer, 'rb'))

    # Get corpus
    def categorize(age: str) -> int:
        if int(age) <= 19:
            return 0
        elif int(age) <= 29:
            return 1
        else:
            return 2
    training_data : DataFrame = None
    for csv_file in os.listdir(args.corpus):
        with open(os.path.join(args.corpus, csv_file), encoding='utf-8') as file:
            data = pd.read_csv(file, names=['bloggerID', 'gender', 'age', 'zodiac', 'blog'])
            if training_data is None:
                training_data = data
            else:
                training_data = training_data.append(data)
    training_data['age'] = training_data['age'].apply(categorize)

    # Apply corpus
    X_train = vectorizer.transform(training_data['blog'])

    # EDIT HERE
    # Train classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=20, n_jobs=-1, multi_class='ovr')

    clf.fit(X_train, training_data[args.label])

    # Save classifier on disk
    save_folder, _ = os.path.split(args.save)
    os.makedirs(save_folder, exist_ok=True)

    with open(args.save, "wb") as model:
        dump(clf, model)


if __name__ == "__main__":
    main()