# Make predictions on the test set
import argparse
import os
from pickle import dump, load
import pandas as pd
from pandas.core.frame import DataFrame
import csv
csv.field_size_limit(1000000)

def get_args():
    parser = argparse.ArgumentParser(description="Makes predictions and saves them to disk")

    parser.add_argument("quiz", help="Quiz file")
    parser.add_argument("model", help="Model kind")
    parser.add_argument("vectorizer", help="Vectorizer kind")
    parser.add_argument("save", help="Save file for the predictions")

    return parser.parse_args()

def main():
    args = get_args()

    # Get vectorizer
    vectorizer = load(open(f"data/vectorizers/{args.vectorizer}.vec", 'rb'))

    # Get models
    age_model = load(open(f"data/models/age-{args.model}-{args.vectorizer}.model", 'rb'))
    gender_model = load(open(f"data/models/gender-{args.model}-{args.vectorizer}.model", 'rb'))
    zodiac_model = load(open(f"data/models/zodiac-{args.model}-{args.vectorizer}.model", 'rb'))

    # Read quiz set
    with open(args.quiz, "r", encoding='utf-8') as file:
        test_data = pd.read_csv(file, names=['bloggerID', 'blog'])

    # Predict
    X_test = vectorizer.transform(test_data['blog'])
    y_age = age_model.predict(X_test)
    y_gender = gender_model.predict(X_test)
    y_zodiac = zodiac_model.predict(X_test)

    # Save predictions
    save_folder, _ = os.path.split(args.save)
    os.makedirs(save_folder, exist_ok=True)

    y_all = DataFrame(columns=["bloggerID", "gender", "age", "zodiac"])
    y_all["bloggerID"] = test_data["bloggerID"]
    y_all["gender"] = y_gender
    y_all["age"] = y_age
    y_all["zodiac"] = y_zodiac

    with open(args.save, "w") as predictions:
        y_all.to_csv(predictions, header=False, index=False)

if __name__ == "__main__":
    main()