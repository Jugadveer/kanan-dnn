"""Feature selection with RFECV: recursive feature elimination, cross-validated,
followed by the one-standard-error rule."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from earlywarning import settings
from earlywarning.encoding import flat_encoder, source_columns

RULES = ("one_standard_error", "best")


def ranking_model() -> LogisticRegression:
    """The model whose coefficients decide which column goes next."""
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def one_standard_error_count(rfecv: RFECV, folds: int) -> int:
    """Fewest encoded columns whose CV score is within one standard error of
    the best score (the "1-SE rule" of Breiman et al., 1984).

    Differences smaller than one standard error are noise, so among the column
    counts that cannot be told apart from the best, the simplest is taken.
    """
    means = rfecv.cv_results_["mean_test_score"]
    stds = rfecv.cv_results_["std_test_score"]
    counts = rfecv.cv_results_["n_features"]
    best = means.argmax()
    return int(counts[means >= means[best] - stds[best] / np.sqrt(folds)].min())


def select_columns(X_train: pd.DataFrame, y_train: np.ndarray,
                   rule: str = settings.CONFIG["selection"]["rule"]) -> dict:
    """Decide which raw columns the model and the app will use.

    How RFECV works:
    1. Fit a Logistic Regression on all encoded columns.
    2. Drop the encoded column with the smallest coefficients (least useful).
    3. Repeat down to min_columns, scoring every column count by
       cross-validated balanced accuracy.
    rule="best" keeps the best-scoring count; rule="one_standard_error" (the
    default) keeps the fewest columns that score as well within noise.

    Logistic Regression works on one-hot columns, so only some values of a text
    column may survive. A raw column stays if any of its encoded columns did.
    """
    if rule not in RULES:
        raise ValueError(f"unknown selection rule {rule!r}; use one of {RULES}")
    selection = settings.CONFIG["selection"]
    encoded = flat_encoder().fit(X_train)
    X_encoded = encoded.transform(X_train)
    rfecv = RFECV(ranking_model(), step=1, min_features_to_select=selection["min_columns"],
                  cv=StratifiedKFold(selection["folds"], shuffle=True, random_state=settings.SEED),
                  scoring="balanced_accuracy").fit(X_encoded, y_train)

    count = rfecv.n_features_ if rule == "best" else one_standard_error_count(rfecv, selection["folds"])
    support = RFE(ranking_model(), n_features_to_select=count, step=1).fit(X_encoded, y_train).support_

    sources = np.array(source_columns(encoded))
    kept = [column for column in X_train.columns if column in set(sources[support])]
    return {
        "kept": kept,
        "dropped": [column for column in X_train.columns if column not in kept],
        "encoded_total": len(sources),
        "best_count": int(rfecv.n_features_),
        "encoded_kept": count,
        # cross-validated balanced accuracy (mean, std) for every number of encoded columns tried
        "scores_by_count": {int(n): [round(float(mean), 4), round(float(std), 4)] for n, mean, std in zip(
            rfecv.cv_results_["n_features"], rfecv.cv_results_["mean_test_score"],
            rfecv.cv_results_["std_test_score"])},
    }
