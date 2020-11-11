# Examine the features of logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="Finds the properties of the logistic regression model")

    parser.add_argument("model", help="The logistic regression model")

    parser.add_argument("vectorizer", help="The vectorizer used by the model")

    return parser.parse_args()

def main():
    args = get_args()

    model : LogisticRegression = load(open(args.model, "rb"))
    vectorizer : TfidfVectorizer = load(open(args.vectorizer, "rb"))

    int_to_feature = vectorizer.get_feature_names()

    # Examine coefficients of the feature matrix
    # binary classification case
    sorted_word_importance = sorted(zip(model.coef_[0], range(len(model.coef_[0]))),\
        key=lambda x: x[0], reverse=True)
    ten_largest = sorted_word_importance[:10]
    ten_smallest = sorted_word_importance[-10:]

    df_largest = pd.DataFrame(ten_largest, columns=["value", "feature"])
    df_largest["word"] = df_largest["feature"].apply(lambda x: int_to_feature[x])
    df_smallest = pd.DataFrame(ten_smallest, columns=["value", "feature"])
    df_smallest["word"] = df_smallest["feature"].apply(lambda x: int_to_feature[x])

    df_largest.to_markdown(open("logistic_words_largest.md", "w"))
    df_smallest.to_markdown(open("logistic_words_smallest.md", "w"))

    # plot bar graphs
    import matplotlib.pyplot as plt
    plt.bar(range(len(model.coef_[0])), model.coef_[0])
    axes = plt.gca()
    axes.set_ylim([-3,3])
    plt.savefig("bar.png")

if __name__ == "__main__":
    main()