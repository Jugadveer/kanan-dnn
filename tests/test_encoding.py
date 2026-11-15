"""Encoders learn only from the rows they are fitted on."""

import numpy as np
import pandas as pd

from earlywarning.encoding import NUMBERS_INPUT, EmbeddingEncoder, OneHotInputs, flat_encoder, source_columns
from earlywarning.loading import inputs_and_target, read_students, three_way_split


def training_split():
    X, y = inputs_and_target(read_students())
    return three_way_split(X, y)


def test_embedding_encoder_statistics_come_from_training_rows_only():
    parts = training_split()
    X_train = parts["train"][0][["G1", "absences", "school", "Mjob"]]
    encoder = EmbeddingEncoder().fit(X_train)
    assert np.allclose(encoder.scaler.mean_, X_train[["G1", "absences"]].mean().to_numpy())
    # transformed training numbers have mean 0 (they would not, if fitted elsewhere)
    assert np.allclose(encoder.transform(X_train)[NUMBERS_INPUT].mean(axis=0), 0, atol=1e-5)


def test_unseen_text_value_gets_code_zero():
    encoder = EmbeddingEncoder().fit(pd.DataFrame({"Mjob": ["teacher", "health"], "age": [16, 17]}))
    codes = encoder.transform(pd.DataFrame({"Mjob": ["teacher", "astronaut"], "age": [16, 16]}))["Mjob"]
    assert codes.tolist()[1] == 0 and codes.tolist()[0] >= 1
    assert encoder.vocabulary_sizes() == {"Mjob": 3}  # two values + "unseen"


def test_source_columns_trace_one_hot_columns_back():
    X = pd.DataFrame({"G1": [10, 12, 8], "Mjob": ["teacher", "health", "other"]})
    fitted = flat_encoder().fit(X)
    assert source_columns(fitted) == ["G1", "Mjob", "Mjob", "Mjob"]


def test_one_hot_inputs_width():
    X = pd.DataFrame({"G1": [10, 12, 8], "Mjob": ["teacher", "health", "other"]})
    encoder = OneHotInputs().fit(X)
    assert encoder.number_count() == 4 and encoder.vocabulary_sizes() == {}
