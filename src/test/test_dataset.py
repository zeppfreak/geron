import pytest

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core.dataset import Dataset

def test_csv_reader_dataset():
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
    filepaths = ['../../data/train.csv']
    dataset = Dataset.csv_reader_dataset(filepaths)

    for d in dataset.take(1):
        print(d)

def test_csv_to_np_dataset():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    filepath = "../../data/train.csv"
    print(filepath)
    d = Dataset.csv_to_np_dataset(filepath)
    print(d)