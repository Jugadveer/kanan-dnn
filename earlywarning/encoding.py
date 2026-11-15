"""Turning raw student rows into numbers, in two forms:

- a flat table (one-hot + standardised) for Logistic Regression, KNN and the
  decision tree, which only accept one matrix of numbers;
- named inputs for the neural network: an integer code per text column (fed
  to an embedding layer) plus one matrix of standardised numbers.

Every encoder learns its codes, means and spreads in fit(), which is only ever
called on the rows a model trains on.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMBERS_INPUT = "numbers"  # name of the network input that holds the numeric columns


def text_columns(X: pd.DataFrame) -> list[str]:
    """Columns stored as words, such as school ('GP'/'MS') or Mjob ('teacher')."""
    return [column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])]


def number_columns(X: pd.DataFrame) -> list[str]:
    """Grades, counts and ordered codes, such as G1, absences or studytime (1-4)."""
    return [column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])]


def flat_encoder() -> ColumnTransformer:
    """Unfitted transformer: standardise numbers, one-hot encode text columns.

    The columns are picked by type when it is fitted, so the same recipe works
    whatever columns feature selection kept.
    """
    return ColumnTransformer([
        ("numbers", StandardScaler(), make_column_selector(dtype_include="number")),
        ("text", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         make_column_selector(dtype_exclude="number")),
    ])


def source_columns(fitted: ColumnTransformer) -> list[str]:
    """The raw column behind every output column of a fitted flat_encoder.

    Numbers map one to one; a text column with 5 values produces 5 one-hot
    columns, all traced back to the same raw column.
    """
    sources = []
    for name, transformer, columns in fitted.transformers_:
        if name == "numbers":
            sources += columns
        elif name == "text":
            for column, values in zip(columns, transformer.categories_):
                sources += [column] * len(values)
    return sources


class EmbeddingEncoder:
    """Prepares the network's inputs: integer codes for embeddings + numbers."""

    def fit(self, X: pd.DataFrame) -> "EmbeddingEncoder":
        self.text = text_columns(X)
        self.numbers = number_columns(X)
        # Unseen values become -1; transform() shifts every code up by one,
        # so 0 means "unseen" and the real values start at 1.
        self.coder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.scaler = StandardScaler()
        if self.text:
            self.coder.fit(X[self.text])
        if self.numbers:
            self.scaler.fit(X[self.numbers])
        return self

    def vocabulary_sizes(self) -> dict[str, int]:
        """Rows each embedding table needs: one per value, plus one for "unseen"."""
        if not self.text:
            return {}
        return {column: len(values) + 1 for column, values in zip(self.text, self.coder.categories_)}

    def number_count(self) -> int:
        return len(self.numbers)

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        inputs = {}
        if self.text:
            codes = self.coder.transform(X[self.text]).astype("int32") + 1
            for position, column in enumerate(self.text):
                inputs[column] = codes[:, position]
        if self.numbers:
            inputs[NUMBERS_INPUT] = self.scaler.transform(X[self.numbers]).astype("float32")
        return inputs


class OneHotInputs:
    """For the ablation without embeddings: every column goes through the flat
    one-hot encoder into the single numbers input."""

    def fit(self, X: pd.DataFrame) -> "OneHotInputs":
        self.flat = flat_encoder().fit(X)
        return self

    def vocabulary_sizes(self) -> dict[str, int]:
        return {}  # no embedding tables

    def number_count(self) -> int:
        return len(self.flat.get_feature_names_out())

    def transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {NUMBERS_INPUT: self.flat.transform(X).astype("float32")}
