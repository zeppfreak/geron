import pytest

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from logic.preprocess import create_preprocess

def test_create_preprocess__not_exist_model():
    model_id = "not_exist_model_id"
    preprocess = create_preprocess(model_id)

    assert preprocess is None