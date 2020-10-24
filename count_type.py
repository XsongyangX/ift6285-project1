"""
Count type

Counts the number of tokens and types inside a given directory,
ignoring all binary files and nested directories.

Produces always one csv file that tracks the number of types compiled
at the end of reading each file. Optionally produces an execution
time csv file.

The vocabulary of the corpus is stored in a json file, by default
'vocabulary.json'.
"""

import argparse
import json
import os
import sys
from typing import Any, Iterator
import subprocess
import time
import pandas as pd
from queue import Queue
import threading

def read_words(path: str) -> Iterator[str]:
    """Reads a file and give an iterator over its words

    Args:
        file (str): File name

    Yields:
        Iterator[str]: Generator over the file's words
    """
    try:
        for line in open_csv(path):
            for word in line.split():
                yield word
    except OSError as error:
        print(error, file=sys.stderr)


def load_csv_files(directory: str) -> Iterator[str]:
    """Returns a generator iterating on the text file names only

    Args:
        path (str): directory of interest on disk

    Yields:
        Iterator[str]: Generator over the file names with full path
    """
    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        # ignore directories
        if os.path.isdir(path):
            continue

        # ignore binary files
        process = subprocess.Popen(
            ['file', '--mime', path], stdout=subprocess.PIPE, text=True)
        mime, error = process.communicate()
        if error is not None:
            print(error, file=sys.stderr)
        if 'charset=binary' in mime:
            continue

        # only take csv
        if file.endswith(".csv"):
            yield path


def verify(args: argparse.Namespace):
    """Checks if the given path exists

    Args:
        args (argparse.Namespace): argparse results

    Raises:
        FileNotFoundError: When the given path is not a directory
    """
    if not os.path.isdir(args.directory):
        raise FileNotFoundError(
            "Directory not found: {}".format(args.directory))


def open_csv(path: str) -> Iterator[str]:
    """Reads the text value of the csv blog line by line

    Args:
        path (str): path to the csv

    Yields:
        str: Text entry of a row in the csv
    """
    if not path.endswith(".csv"):
        raise Exception("Not a csv file encountered in open_csv: {}".format(path))
    if Timer.is_timing:
        Timer.get_current_instance().log()
    # with open(path, mode='r') as csv_file:
    #     blogreader = csv.reader(csv_file)
    #     for row in blogreader:
    #         yield row[-1]
    dataframe = pd.read_csv(path, names=('ID', 'Gender', 'Age', 'Zodiac', 'Blog'))
    for row in dataframe['Blog']:
        yield row

class Logger(object):
    __instance = None

    @staticmethod
    def get_current_instance():
        """
        Get a singleton instance of the logger
        """
        if Logger.__instance is None:
            raise Exception("No Logger")
        return Logger.__instance

    def __init__(self, path: str):
        """Creates a logger instance

        Args:
            path (str): Path to the file where to put the log
        """
        self.path_to_log = path

        # clean up the log file
        open(self.path_to_log, 'w').close()

        if Logger.__instance is None and type(self) is Logger:
            Logger.__instance = self

        self.queue : Queue = Queue()
        self.set_daemon()

    def set_daemon(self):

        # multithreading
        def background_logger():
            while True:
                item = self.queue.get()
                with open(self.path_to_log, 'a') as time_log:
                    time_log.write("{}\n".format(item))
                self.queue.task_done()
        threading.Thread(target=background_logger, daemon=True,\
            name="Background {}".format(type(self))).start()


    def log(self, message: Any):
        # enqueue the request
        self.queue.put(message) # dummy value
    
    def block_until_logged(self):
        self.queue.join()
        
class Timer(Logger):
    is_timing = False
    __instance = None

    @staticmethod
    def get_current_instance():
        """
        Get a singleton instance of the logger
        """
        if Timer.__instance is None:
            raise Exception("No Timer")
        return Timer.__instance

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.start = time.time()
        Timer.is_timing = True
        if Timer.__instance is None:
            Timer.__instance = self

    def log(self):
        # enqueue the request
        super().log(time.time() - self.start)

def parse() -> argparse.Namespace:
    """Parses arguments from command line
    """
    parser = argparse.ArgumentParser(
        description="""Counts the number of token types in the directory\'s files 
        and stores the result inside a folder named data""")

    parser.add_argument(
        'directory', help='Directory in which to operate the counting')

    parser.add_argument('--time',
                        help="""Whether to time the counting as well.
        If so, the time is store in the file time_count_type.csv""",
                        const=True, action='store_const', default=False)

    parser.add_argument('--count', metavar='log',
                        help="""Name of the file for
            the token type count seen so far accumulated per file (default: types_per_file.csv)""",
                        default='count_type.csv')

    parser.add_argument('--json', metavar='vocabulary',
                        help="""Name of the json of the vocabulary
                        (default: vocabulary.json)""",
                        default='vocabulary.json')

    return parser.parse_args()


def main():
    """Parse arguments, load files, read and compile tokens and types
    """

    args = parse()
    verify(args)
    files = load_csv_files(args.directory)

    # initiate log files for time and count
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if args.time:
        Timer("data/{}".format("time_count_type.csv"))
    Logger("data/{}".format(args.count if args.count is not None else "types_per_file.csv"))

    # initiate empty dictionary and counter
    vocabulary = dict()
    count = 0
    types = 0

    # populate dictionary
    for file in files:
        for word in read_words(file):
            count += 1
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1
                types += 1

        Logger.get_current_instance().log(types)

    # produce a json vocabulary
    with open("data/{}".format(args.json if args.json is not None else "vocabulary.json"), 'w') as voc_file:
        json_representation = json.dumps(vocabulary)
        voc_file.write(json_representation)

    # final message
    print(
        """Finished counting:
        {word_count} words in total
        {type_count} types in total
        """.format(word_count=count, type_count=types))


if __name__ == "__main__":
    main()
