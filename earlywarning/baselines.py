"""The models the network is compared with.

- Two trivial classifiers: what any model must beat.
- A naive rule, "the group of the first-term grade": what the network must
  beat to be worth having at all, since G1 alone already says a lot.
- Two traditional machine-learning models, K-Nearest Neighbours and a decision
  tree, with search spaces from config.yaml.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from earlywarning import settings
from earlywarning.encoding import flat_encoder
from earlywarning.loading import category_of


class FirstTermGroup:
    """Naive rule: predict the group the first-term grade G1 falls into
    (G1 of 0-9 -> Needs support, 10-13 -> On track, 14-20 -> High achiever).
    It learns nothing, so fit() does nothing."""

    uses_selected_columns = False  # always reads G1, whatever feature selection kept

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FirstTermGroup":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.eye(len(settings.CATEGORIES))[category_of(X["G1"])]


def trivial_models() -> dict:
    """Most frequent: always "On track". Stratified: random guesses in the
    training proportions. Both ignore the student entirely."""
    return {
        "Always the most common group": DummyClassifier(strategy="most_frequent"),
        "Random guess (stratified)": DummyClassifier(strategy="stratified", random_state=settings.SEED),
    }


def knn(params: dict) -> Pipeline:
    """A student gets the most common group among the k most similar students
    (Euclidean distance on standardised + one-hot columns)."""
    return Pipeline([("encode", flat_encoder()), ("model", KNeighborsClassifier(**params))])


def decision_tree(params: dict) -> Pipeline:
    """Yes/no questions about the columns, each chosen to make the groups purer
    (lower Gini impurity). Groups are weighted against their rarity."""
    return Pipeline([("encode", flat_encoder()),
                     ("model", DecisionTreeClassifier(class_weight="balanced",
                                                      random_state=settings.SEED, **params))])


# family name -> (builder, its search space in config.yaml)
TRADITIONAL = {
    "K-Nearest Neighbours": (knn, settings.CONFIG["baselines"]["knn"]),
    "Decision Tree": (decision_tree, settings.CONFIG["baselines"]["tree"]),
}


def search_space(family: str) -> list[dict]:
    """Every combination of the family's settings (a grid)."""
    return list(ParameterGrid(TRADITIONAL[family][1]))
