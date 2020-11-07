
"""
Sets up a preprocessing pipeline
"""

import argparse
import os
import threading
from typing import Callable, Iterator, List, Tuple, Union
from queue import Queue
import pandas as pd
from pandas.core.frame import DataFrame


class Labels():
    """Represents the labels of a data point (a blog)
    """

    def __init__(self, id: str, gender: str, age: int, zodiac: str):
        """Reads the labels of the blog post and categorizes as needed

        Args:
            gender (str): Gender can only be male or female
            age (int): Age of the author
            zodiac (str): One of the twelve zodiac signs
        """
        self.id = id
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
        return "(ID: {}, Gender: {}, Age: {}, Zodiac: {})"\
            .format(self.id, self.gender, self.age, self.zodiac)


class Preprocessor():
    """
    Stores settings of a preprocessing pipeline
    """

    def __init__(self, path_to_corpus_directory: str):
        """Initiates a preprocessing pipeline for the corpus at the given directory

        Args:
            path_to_corpus_directory (str): Directory of the corpus
        """

        self.path_to_corpus_directory = path_to_corpus_directory

        self.preprocesses: List[Callable[[List[str]], List[str]]] = []
        self.tokenizer: Callable[[str], List[str]] = lambda x: x.split()

    def blog_stream(self, return_unparsed_labels: bool = False) -> Iterator[Union[Tuple[str, Labels], DataFrame]]:
        """Iterates over the corpus directly from disk
        """
        # Use multithreading queue
        dataframes: Queue[DataFrame] = Queue(maxsize=5)

        def produce_dataframes():
            for path in os.listdir(self.path_to_corpus_directory):
                if not path.endswith(".csv"):
                    continue
                dataframe = pd.read_csv(os.path.join(self.path_to_corpus_directory, path),
                                        names=('ID', 'Gender', 'Age', 'Zodiac', 'Blog'))
                dataframes.put(dataframe)
        producer = threading.Thread(
            target=produce_dataframes, name="csv reader")
        producer.start()

        while producer.is_alive() or dataframes.qsize() != 0:
            dataframe = dataframes.get()
            dataframes.task_done()
            if return_unparsed_labels:
                yield dataframe
            else:
                for row in dataframe['Blog']:
                    yield row, Labels(dataframe['ID'][0], dataframe['Gender'][0], dataframe['Age'][0], dataframe['Zodiac'][0])

    def save(self, directory: str):
        """Saves the preprocessed corpus to the directory

        Args:
            directory (str): Folder where to save
        """
        import shutil
        shutil.rmtree(directory)
        os.makedirs(directory)

        processed_dataframes : Queue[DataFrame] = Queue()

        def csv_writer():
            while True:
                dataframe = processed_dataframes.get()
                dataframe.to_csv(
                    "{folder}/{ID}.{gender}.{age}.{zodiac}.csv"
                    .format(folder=directory, ID=dataframe['ID'][0], gender=dataframe['Gender'][0],
                        age=dataframe['Age'][0], zodiac=dataframe['Zodiac'][0]),
                    header=False, index=False)
                processed_dataframes.task_done()
        writer = threading.Thread(target=csv_writer, name="csv writer", daemon=True)
        writer.start()

        for dataframe in self.blog_stream(True):
            dataframe['Blog'] = dataframe['Blog'].apply(
                lambda blog: " ".join(self.preprocess(blog)))
            processed_dataframes.put(dataframe)

        processed_dataframes.join()
        

    def preprocess(self, blog: str) -> List[str]:
        """Preprocess a string with the instance's tokenizer
        and preprocessing procedures

        Args:
            s (str): A string to be preprocessed

        Returns:
            List[str]: Preprocessed tokens
        """
        preprocessed = self.tokenizer(blog)
        for procedure in self.preprocesses:
            preprocessed = procedure(preprocessed)
        return preprocessed

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
            yield self.preprocess(blog), blogger


def main():
    parser = argparse.ArgumentParser("Runs the full preprocessing pipeline")

    parser.add_argument("corpus", help="Where the corpus is located")

    parser.add_argument("save", metavar="save-location", help="Where to save")

    args = parser.parse_args()

    preprocessor = Preprocessor(args.corpus)
    # choose a tokenizer, by default it is python's split
    from nltk.tokenize import TweetTokenizer
    tweet = TweetTokenizer(preserve_case=False, reduce_len=True)
    preprocessor.tokenizer = tweet.tokenize  # a function

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
