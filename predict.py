# Make predictions on the test set
import argparse
import os
from pickle import dump, load
import pandas as pd
from pandas.core.frame import DataFrame

def get_args():
    parser = argparse.ArgumentParser(description="Makes predictions and saves them to disk")

    parser.add_argument("test", help="Directory of the test slice in the blog csv corpus")
    parser.add_argument("model", help="Model file")
    parser.add_argument("vectorizer", help="Vectorizer file")
    parser.add_argument("save", help="Save location for the predictions")

    parser.add_argument("--label", help="Which label to predict, ex. 'gender' (default), 'age', 'zodiac'",\
        default="gender")

    return parser.parse_args()

def main():
    args = get_args()

    # Get vectorizer
    vectorizer = load(open(args.vectorizer, 'rb'))

    # Get model
    model = load(open(args.model, 'rb'))

    # Read test set
    def categorize(age: str) -> int:
        if int(age) <= 19:
            return 0
        elif int(age) <= 29:
            return 1
        else:
            return 2
    test_data : DataFrame = None
    for csv_file in os.listdir(args.test):
        with open(os.path.join(args.test, csv_file), encoding='utf-8') as file:
            data = pd.read_csv(file, names=['bloggerID', 'gender', 'age', 'zodiac', 'blog'])
            if test_data is None:
                test_data = data
            else:
                test_data = test_data.append(data)
    test_data['age'] = test_data['age'].apply(categorize)

    # Predict
    X_test = vectorizer.transform(test_data['blog'])
    y = model.predict(X_test)

    # Save predictions
    save_folder, _ = os.path.split(args.save)
    os.makedirs(save_folder, exist_ok=True)

    with open(args.save, "wb") as predictions:
        dump([test_data[args.label], y], predictions)

if __name__ == "__main__":
    main()