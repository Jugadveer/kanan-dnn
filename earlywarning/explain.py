"""Explanations: which inputs the network relies on overall (permutation
importance) and what pushed one student's prediction (reset-to-typical)."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from earlywarning import settings


def permutation_importance(predict: Callable[[pd.DataFrame], np.ndarray],
                           X_test: pd.DataFrame, y_test: np.ndarray) -> list[dict]:
    """Shuffle one column at a time and measure how much balanced accuracy drops.

    Shuffling breaks the link between that column and the student while keeping
    its values realistic. A big drop means the model depends on the column; a
    drop near zero (or below) means it barely uses it. Each column is shuffled
    several times because one shuffle can be lucky.

    predict maps raw student rows to category probabilities.
    """
    rng = np.random.default_rng(settings.SEED)
    shuffles = settings.CONFIG["explain"]["shuffles"]
    baseline = balanced_accuracy_score(y_test, predict(X_test).argmax(axis=1))

    rows = []
    for column in X_test.columns:
        drops = []
        for _ in range(shuffles):
            shuffled = X_test.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            score = balanced_accuracy_score(y_test, predict(shuffled).argmax(axis=1))
            drops.append(baseline - score)
        rows.append({"column": column, "mean_drop": float(np.mean(drops)),
                     "std_drop": float(np.std(drops))})

    return sorted(rows, key=lambda row: row["mean_drop"], reverse=True)


def what_pushed(predict: Callable[[pd.DataFrame], np.ndarray], student: dict, typical: dict) -> list[dict]:
    """For one student: how much each answer moved the predicted group's probability.

    Each input that differs from the typical answer is reset to it, one at a
    time; the change in the predicted group's probability is that answer's
    effect. Positive: the answer pushed towards the prediction. It explains
    single answers, not combinations of them.
    """
    probabilities = predict(pd.DataFrame([student]))[0]
    group = int(probabilities.argmax())
    different = [column for column in student if student[column] != typical[column]]
    if not different:
        return []
    reset_rows = pd.DataFrame([{**student, column: typical[column]} for column in different])
    without = predict(reset_rows)[:, group]
    effects = [{"column": column, "answer": student[column], "typical": typical[column],
                "effect": float(probabilities[group] - probability)}
               for column, probability in zip(different, without)]
    return sorted(effects, key=lambda row: abs(row["effect"]), reverse=True)
