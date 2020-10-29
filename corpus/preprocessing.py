
"""
Sets up a preprocessing pipeline
"""

from typing import Callable, Iterator, List, Tuple
import os
import pandas as pd
import argparse

from pandas.core.frame import DataFrame


class Labels(object):
    """Represents the labels of a data point (a blog)
    """
    def __init__(self, gender: str, age: int, zodiac: str):
        """Reads the labels of the blog post and categorizes as needed

        Args:
            gender (str): Gender can only be male or female
            age (int): Age of the author
            zodiac (str): One of the twelve zodiac signs
        """
        self.gender = gender.lower()
        def age_categories(age: int) -> int:
            age = int(age)
            if 0 <= age <= 19:
                return 0
            elif 20 <= age <= 29:
                return 1
            elif 30 <= age:
                return 2
            else:
                raise AttributeError("Age {} is not allowed".format(age))
        self.age = age_categories(age)
        self.zodiac = zodiac

    def __str__(self) -> str:
        return "(Gender: {}, Age: {}, Zodiac: {})".format(self.gender, self.age, self.zodiac)

class Preprocessor(object):
    """
    Stores settings of a preprocessing pipeline
    """
    
    def __init__(self, path_to_corpus_directory: str):
        """Initiates a preprocessing pipeline for the corpus at the given directory

        Args:
            path_to_corpus_directory (str): Directory of the corpus
        """
        
        self.path_to_corpus_directory = path_to_corpus_directory

        self.preprocesses : List[Callable[[List[str]], List[str]]] = []
        self.tokenizer: Callable[[str], List[str]] = lambda x : x.split()

    def blog_stream(self, return_unparsed_labels: bool = False):
        """Iterates over the corpus directly from disk
        """
        for path in os.listdir(self.path_to_corpus_directory):
            if not path.endswith(".csv"):
                continue
            dataframe = pd.read_csv(os.path.join(self.path_to_corpus_directory, path),\
                names=('ID', 'Gender', 'Age', 'Zodiac', 'Blog'))
            for row in dataframe['Blog']:
                if return_unparsed_labels:
                    yield row, dataframe['ID'][0], dataframe['Gender'][0], dataframe['Age'][0], dataframe['Zodiac'][0]
                else:
                    yield row, Labels(dataframe['Gender'][0], dataframe['Age'][0], dataframe['Zodiac'][0])

    def save(self, directory: str):
        """Saves the preprocessed corpus to the directory

        Args:
            directory (str): Folder where to save
        """
        import shutil
        shutil.rmtree(directory)
        os.makedirs(directory)
        for blog, id, gender, age, zodiac in self.blog_stream(True):
            preprocessed = self.tokenizer(blog)
            for preprocess in self.preprocesses:
                preprocessed = preprocess(preprocessed)
            dataframe = DataFrame([[id, gender, age, zodiac, " ".join(preprocessed)]])
            with open(os.path.join(directory, "{id}.{gender}.{age}.{zodiac}.csv".format(id=id, gender=gender, age=age, zodiac=zodiac)), 'a+') as file:
                dataframe.to_csv(file, header=False, index=False, mode='a+', line_terminator='\n')


    def run(self) -> List[Tuple[List[str], Labels]]:
        """
        Runs the preprocessor on the corpus and returns the corpus all at once.

        Returns:
            List of data points: a data point is the tuple (tokens, labels)
        """
        return list(self.run_yield())

    def run_yield(self) -> Iterator[Tuple[List[str], Labels]]:
        """Runs the preprocessor on the corpus and yields data point by data point

        Yields:
            Iterator[Tuple[List[str], Labels]]: Iterator over the data points
        """
        for blog, blogger in self.blog_stream():
            preprocessed = self.tokenizer(blog)
            for preprocess in self.preprocesses:
                preprocessed = preprocess(preprocessed)
            yield preprocessed, blogger

def main():
    parser = argparse.ArgumentParser("Runs the full preprocessing pipeline")

    parser.add_argument("corpus", help="Where the corpus is located")

    parser.add_argument("save", metavar="save-location" , help="Where to save")

    args = parser.parse_args()

    preprocessor = Preprocessor(args.corpus)
    # choose a tokenizer, by default it is python's split
    from nltk.tokenize import TweetTokenizer
    tweet = TweetTokenizer(preserve_case=False, reduce_len=True)
    preprocessor.tokenizer = tweet.tokenize # a function

    # append as many non-tokenizing preprocesses as needed, e.g. lowercase mapping
    # fn( List[str] ) -> List[str]

    def map_numbers(words: List[str]):
        for word in words:
            yield 'NUM' if word.isnumeric() else word

    def map_nonascii(words: List[str]):
        for word in words:
            yield 'NONASCII' if not word.isascii() else word

    # stemming is too weird
    # from nltk.stem import PorterStemmer
    # stemmer = PorterStemmer()

    preprocessor.preprocesses.extend([
        # lowercase is handled by the tweet tokenizer
        # numbers
        lambda x: list(map_numbers(x)),
        # nonascii
        lambda x: list(map_nonascii(x))
        # stemming
        #lambda x: [stemmer.stem(word) for word in x]
        ])

    # run the preprocessing pipeline
    preprocessor.save(args.save)
        

if __name__ == "__main__":
    main()