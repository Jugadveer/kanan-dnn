"""Scores, confidence intervals and significance tests.

Three questions, from the most detailed to the one a teacher asks first:
1. Which of the three groups? (balanced accuracy, macro F1, ROC-AUC)
2. Does this student need support, yes or no? (recall, precision, ROC-AUC)
3. Is the prediction at most one group off? (within-one accuracy)
"""

import numpy as np
from scipy.stats import binomtest, ttest_rel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from earlywarning import settings

NEEDS_SUPPORT = 0  # index of the group the early warning is about

# The metrics reported first everywhere, with the words used for them.
HEADLINES = {
    "support_roc_auc": "Needs support? Ranking of students by risk (ROC-AUC)",
    "support_recall": "Needs support? Share of them found (recall)",
    "support_precision": "Needs support? Share of flagged students who do (precision)",
    "roc_auc": "Which group? Ranking by probability (ROC-AUC, one-vs-rest)",
    "balanced_accuracy": "Which group? Balanced accuracy",
    "macro_f1": "Which group? Macro F1",
    "within_one": "At most one group off",
}


def score(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Every metric reported for a model on one set of students."""
    predicted = probabilities.argmax(axis=1)
    flagged, needs_support = predicted == NEEDS_SUPPORT, y_true == NEEDS_SUPPORT
    return {
        # three groups
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, probabilities, multi_class="ovr")),
        "accuracy": float(accuracy_score(y_true, predicted)),
        # Needs support, yes or no
        "support_recall": float(recall_score(needs_support, flagged, zero_division=0)),
        "support_precision": float(precision_score(needs_support, flagged, zero_division=0)),
        "support_roc_auc": float(roc_auc_score(needs_support, probabilities[:, NEEDS_SUPPORT])),
        "support_accuracy": float(accuracy_score(needs_support, flagged)),
        # never two groups off (Needs support <-> High achiever)
        "within_one": float(np.mean(np.abs(predicted - y_true) <= 1)),
    }


def bootstrap_intervals(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, list[float]]:
    """95 % confidence interval for every metric by the percentile bootstrap.

    Draw the test students with replacement (same size), score, repeat 2000
    times; the middle 95 % of the scores is the interval. It shows how much a
    score could move with a different sample of 98 students.
    """
    evaluation = settings.CONFIG["evaluation"]
    rng = np.random.default_rng(settings.SEED)
    samples = []
    for _ in range(evaluation["bootstrap_resamples"]):
        rows = rng.integers(0, len(y_true), len(y_true))
        samples.append(score(y_true[rows], probabilities[rows]))
    tail = (1 - evaluation["confidence"]) / 2 * 100
    return {metric: np.percentile([sample[metric] for sample in samples], [tail, 100 - tail]).round(4).tolist()
            for metric in samples[0]}


def mcnemar(y_true: np.ndarray, predicted_a: np.ndarray, predicted_b: np.ndarray) -> dict:
    """Exact McNemar test: do two models make different numbers of mistakes
    on the same students?

    Only the students where exactly one model is right count: b (A right, B
    wrong) and c (A wrong, B right). If the models were equally good, each
    such student would be a coin flip, so b follows Binomial(b + c, 0.5).
    """
    a_right, b_right = predicted_a == y_true, predicted_b == y_true
    only_a, only_b = int(np.sum(a_right & ~b_right)), int(np.sum(~a_right & b_right))
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_right": only_a, "only_second_right": only_b, "p_value": float(p_value)}


def paired_folds(scores_a: list[float], scores_b: list[float]) -> dict:
    """Compare two models fold by fold on the same cross-validation folds.

    The paired t-test treats folds as independent, which they are not (their
    training sets overlap), so its p-value is optimistic (Dietterich, 1998);
    the count of folds each model wins is reported next to it.
    """
    differences = np.array(scores_a) - np.array(scores_b)
    return {
        "mean_difference": float(differences.mean()),
        "first_wins": int(np.sum(differences > 0)),
        "second_wins": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
        "t_test_p_value": float(ttest_rel(scores_a, scores_b).pvalue),
    }


def per_category(y_true: np.ndarray, probabilities: np.ndarray) -> str:
    """Precision, recall and F1 for each group, as printable text."""
    return classification_report(y_true, probabilities.argmax(axis=1),
                                 target_names=settings.CATEGORIES, zero_division=0)
