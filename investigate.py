# Examine the features of logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from pickle import load
import argparse

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
    for i, label in enumerate(model.classes_):
        print(label, "top features from logistic regression")
        word_importance = [abs(x) for x in model.coef_[i]]
        sorted_word_importance = sorted(zip(word_importance, range(len(model.coef_[i]))),\
            key=lambda x: x[0], reverse=True)
        words = [int_to_feature[word] for _,word in sorted_word_importance[:10]]
        print("\t".join(words))

    # plot bar graphs
    import matplotlib.pyplot as plt
    plt.bar(range(len(model.coef_[0])), model.coef_[0])
    axes = plt.gca()
    axes.set_ylim([-3,3])
    plt.savefig("bar.png")

if __name__ == "__main__":
    main()