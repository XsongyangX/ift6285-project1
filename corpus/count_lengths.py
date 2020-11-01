from typing import Dict
from preprocessing import Preprocessor
import argparse
import os, json

def parse():
    parser = argparse.ArgumentParser(description="Count the lengths of blogs")
    
    parser.add_argument('directory', help="Where the corpus is located")

    parser.add_argument('results', help="Where to put the counting results")

    return parser.parse_args()

def main():
    args = parse()

    corpus = Preprocessor(args.directory).run()

    lengths : Dict[str, int] = {"<50":0, "50-99":0, "100-199":0, "200+":0}

    for blog, _ in corpus:
        word_count = len(blog)
        if word_count <= 50:
            lengths["<50"] += 1
        elif word_count <= 99:
            lengths["50-99"] += 1
        elif word_count <= 199:
            lengths["100-199"] += 1
        else:
            lengths["200+"] += 1
    
    with open(os.path.join(args.results, "lengths.json"), "w") as file:
        json.dump(lengths, file, indent=4, ensure_ascii=True)


if __name__ == "__main__":
    main()