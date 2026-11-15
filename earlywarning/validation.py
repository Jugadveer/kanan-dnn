"""Cross-validation without selection leakage, shared by every model choice.

Model selection works on the pool of training + validation students (551).
The pool is cut into 5 stratified folds once; in every fold, RFECV runs on
that fold's training part only, so the fold's held-out students never
influence which columns are used. Every model is scored on the same folds,
which also makes per-fold comparisons between models paired.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from earlywarning import settings
from earlywarning.evaluate import score
from earlywarning.selection import select_columns


@dataclass
class Fold:
    train_rows: np.ndarray
    held_out_rows: np.ndarray
    columns: list[str]  # what RFECV kept on train_rows (or every column)


def plan_folds(X: pd.DataFrame, y: np.ndarray, select: bool = True, repeat: int = 0,
               rule: str = settings.CONFIG["selection"]["rule"]) -> list[Fold]:
    """The 5 folds, each with its own feature selection (select=False keeps all
    columns). repeat > 0 shuffles differently, for repeated cross-validation;
    rule is the selection rule (see selection.py)."""
    splitter = StratifiedKFold(settings.CONFIG["model_selection"]["folds"], shuffle=True,
                               random_state=settings.SEED + repeat)
    folds = []
    for train_rows, held_out_rows in splitter.split(X, y):
        columns = (select_columns(X.iloc[train_rows], y[train_rows], rule)["kept"] if select
                   else list(X.columns))
        folds.append(Fold(train_rows, held_out_rows, columns))
    return folds


def columns_for(model, fold: Fold, X: pd.DataFrame) -> list[str]:
    """Most models see only the fold's selected columns; a rule that needs a
    specific column (the naive G1 rule) says so with uses_selected_columns."""
    return fold.columns if getattr(model, "uses_selected_columns", True) else list(X.columns)


def cross_validate(make_model: Callable[[], object], X: pd.DataFrame, y: np.ndarray,
                   folds: list[Fold]) -> list[dict]:
    """Every metric of a fresh model on each fold's held-out students."""
    fold_scores = []
    for fold in folds:
        model = make_model()
        columns = columns_for(model, fold, X)
        model.fit(X.iloc[fold.train_rows][columns], y[fold.train_rows])
        probabilities = model.predict_proba(X.iloc[fold.held_out_rows][columns])
        fold_scores.append(score(y[fold.held_out_rows], probabilities))
    return fold_scores


def metric(fold_scores: list[dict], name: str = "balanced_accuracy") -> list[float]:
    """One metric across folds, e.g. the 5 balanced accuracies."""
    return [scores[name] for scores in fold_scores]


def summarise(fold_scores: list[dict]) -> dict:
    """Mean and standard deviation of every metric over the folds."""
    return {name: {"mean": float(np.mean(metric(fold_scores, name))),
                   "std": float(np.std(metric(fold_scores, name)))}
            for name in fold_scores[0]}
