import argparse
import os
from typing import List, Tuple
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="Finds the best classifier of each metric")

    parser.add_argument("reports", help="Location of the reports")

    return parser.parse_args()

def main():
    args = get_args()

    accuracies : List[Tuple[float, str]] = []
    f1s : List[Tuple[float, str]] = []

    for log in os.listdir(args.reports):
        if not log.endswith(".log"):
            continue

        for line in open(os.path.join(args.reports, log), "r"):
            line = line.split()
            # get accuracy
            if "accuracy" in line:
                accuracies.append((float(line[1]), log))
            # get macro average of f1-scores
            elif "macro" in line:
                f1s.append((float(line[4]), log))

    accuracies.sort(reverse=True, key=lambda x: x[0])
    f1s.sort(reverse=True, key=lambda x: x[0])

    df_accuracies = pd.DataFrame(accuracies, columns=["Accuracy", "Classifier"])
    df_f1s = pd.DataFrame(f1s, columns=["f1", "Classifier"])

    df_accuracies.to_markdown(open("accuracies.md", "w"))
    df_f1s.to_markdown(open("f1s.md", "w"))

if __name__ == "__main__":
    main()