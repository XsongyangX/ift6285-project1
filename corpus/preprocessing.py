
"""
Sets up a preprocessing pipeline
"""

from typing import Callable, Iterator, List, Tuple
import os
import pandas as pd
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

    def blog_stream(self) -> Iterator[Tuple[str, Labels]]:
        """Iterates over the corpus directly from disk
        """
        for path in os.listdir(self.path_to_corpus_directory):
            if not path.endswith(".csv"):
                continue
            dataframe = pd.read_csv(os.path.join(self.path_to_corpus_directory, path),\
                names=('ID', 'Gender', 'Age', 'Zodiac', 'Blog'))
            for row in dataframe['Blog']:
                yield row, Labels(dataframe['Gender'][0], dataframe['Age'][0], dataframe['Zodiac'][0])

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
    preprocessor = Preprocessor("data/excerpt")
    corpus = preprocessor.run()
    for blog, blogger in corpus:
        print(blog[:5], str(blogger))
        

if __name__ == "__main__":
    main()