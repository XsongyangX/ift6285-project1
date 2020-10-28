"""
Count how many different kinds of labels there are
"""

import argparse
import subprocess
from typing import Dict
from preprocessing import Labels
import os, json

def parse():
    parser = argparse.ArgumentParser(description="Count how many kinds of labels there are")
    
    parser.add_argument('directory', help="Where the corpus is located")

    parser.add_argument('results', help="Where to put the counting results")

    return parser.parse_args()

def increment_key(key, dictionary: Dict, inc:int=1):
    if key in dictionary:
        dictionary[key] += inc
    else:
        dictionary[key] = inc

def main():

    args = parse()

    genders : Dict[str, int] = {}
    ages : Dict[int, int] = {}
    zodiacs : Dict[str, int] = {}

    for file in os.listdir(args.directory):
        if not file.endswith(".csv"):
            continue
        
        # parse labels
        _, gender, age, zodiac, _ = file.split('.')
        labels = Labels(gender, age, zodiac)

        # figure multiplicity
        wc = subprocess.run(["wc", os.path.join(args.directory, file), "-l"], capture_output=True, encoding="utf-8")
        results = wc.stdout
        number_of_blogs, _ = results.split()
        number_of_blogs = int(number_of_blogs)

        increment_key(labels.gender, genders, number_of_blogs)
        increment_key(labels.age, ages, number_of_blogs)
        increment_key(labels.zodiac, zodiacs, number_of_blogs)
    
    with open(os.path.join(args.results, "labels.json"), "w") as file:
        all_labels = {"genders": genders, "ages": ages, "zodiacs": zodiacs}
        json.dump(all_labels, file, indent=4, ensure_ascii=True)


if __name__ == "__main__":
    main()