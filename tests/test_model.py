"""The network, calibration, the naive rule and the statistics, on tiny runs."""

import numpy as np
import pandas as pd
import pytest

from earlywarning import settings
from earlywarning.baselines import FirstTermGroup
from earlywarning.calibration import fit_temperature, scale_probabilities
from earlywarning.evaluate import mcnemar, score
from earlywarning.loading import inputs_and_target, read_students, three_way_split
from earlywarning.model import NetworkClassifier, class_alpha

TINY = {"embedding_size": 2, "units": [8], "dropout": 0.1, "l2": 0.001, "learning_rate": 0.01}


@pytest.fixture
def few_epochs(monkeypatch):
    monkeypatch.setitem(settings.CONFIG["training"], "max_epochs", 3)


@pytest.fixture(scope="module")
def small_data():
    X, y = inputs_and_target(read_students())
    (X_train, y_train), (X_val, y_val), _ = three_way_split(X, y).values()
    columns = ["G1", "failures", "school", "higher"]
    return X_train[columns], y_train, X_val[columns], y_val


@pytest.mark.usefixtures("few_epochs")
def test_probabilities_sum_to_one_for_three_groups(small_data):
    X_train, y_train, X_val, _ = small_data
    probabilities = NetworkClassifier(TINY).fit(X_train, y_train).predict_proba(X_val)
    assert probabilities.shape == (len(X_val), 3)
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)


@pytest.mark.usefixtures("few_epochs")
def test_same_seed_gives_identical_predictions(small_data):
    X_train, y_train, X_val, _ = small_data
    first = NetworkClassifier(TINY).fit(X_train, y_train).predict_proba(X_val)
    second = NetworkClassifier(TINY).fit(X_train, y_train).predict_proba(X_val)
    assert np.array_equal(first, second)


def test_class_alpha_weights_rare_groups_more():
    alpha = class_alpha(np.array([0] * 10 + [1] * 60 + [2] * 30))
    assert alpha[0] > alpha[2] > alpha[1]


def test_temperature_keeps_the_predicted_group_and_one_is_identity():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    assert np.allclose(scale_probabilities(probabilities, 1.0), probabilities)
    softer = scale_probabilities(probabilities, 2.0)
    assert (softer.argmax(axis=1) == probabilities.argmax(axis=1)).all()
    assert softer.max() < probabilities.max()


def test_fit_temperature_softens_overconfident_probabilities():
    y = np.array([0, 1, 2, 0, 1, 2])
    overconfident = np.array([[0.98, 0.01, 0.01], [0.98, 0.01, 0.01], [0.01, 0.01, 0.98],
                              [0.01, 0.98, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]])
    assert fit_temperature(overconfident, y) > 1


def test_naive_rule_copies_the_first_term_group():
    probabilities = FirstTermGroup().predict_proba(pd.DataFrame({"G1": [5, 9, 10, 13, 14]}))
    assert probabilities.argmax(axis=1).tolist() == [0, 0, 1, 1, 2]


def test_mcnemar_counts_and_symmetry():
    y = np.array([0, 1, 2, 1, 1, 0])
    result = mcnemar(y, np.array([0, 1, 2, 0, 0, 0]), np.array([1, 1, 2, 1, 1, 0]))
    assert result["only_first_right"] == 1 and result["only_second_right"] == 2
    assert 0 < result["p_value"] <= 1


def test_score_reports_all_three_questions():
    y = np.array([0, 1, 2, 1])
    probabilities = np.eye(3)[[0, 1, 1, 1]] * 0.8 + 0.2 / 3
    scores = score(y, probabilities)
    assert scores["support_recall"] == 1.0 and scores["within_one"] == 1.0
    assert set(scores) >= {"balanced_accuracy", "macro_f1", "roc_auc", "support_precision"}
