"""Feature selection: the one-standard-error rule and leakage-free folds."""

from types import SimpleNamespace

import numpy as np
import pytest

from earlywarning.selection import one_standard_error_count, select_columns


def fake_rfecv(means, stds):
    return SimpleNamespace(cv_results_={"mean_test_score": np.array(means),
                                        "std_test_score": np.array(stds),
                                        "n_features": np.arange(1, len(means) + 1)})


def test_one_standard_error_rule_takes_fewest_columns_within_noise():
    # best is 4 columns (0.80); one standard error = 0.05 / sqrt(5) = 0.022
    rfecv = fake_rfecv([0.70, 0.785, 0.79, 0.80, 0.78], [0.05] * 5)
    assert one_standard_error_count(rfecv, folds=5) == 2


def test_one_standard_error_rule_keeps_best_when_nothing_is_close():
    rfecv = fake_rfecv([0.50, 0.60, 0.80], [0.01] * 3)
    assert one_standard_error_count(rfecv, folds=5) == 3


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        select_columns(None, None, rule="largest")
