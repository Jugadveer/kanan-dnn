"""The saved model and the app's form agree with each other.
Needs `python -m earlywarning train` to have run; skipped (with a reason) otherwise."""

import numpy as np
import pytest

from earlywarning import settings
from earlywarning.encoding import NUMBERS_INPUT
from earlywarning.predictor import RiskPredictor, artifacts_exist

pytestmark = pytest.mark.skipif(not artifacts_exist(), reason="no trained model: run python -m earlywarning train")


@pytest.fixture(scope="module")
def predictor():
    return RiskPredictor()


def test_form_asks_exactly_for_the_model_inputs(predictor):
    form_columns = {field["column"] for field in predictor.form}
    assert form_columns == set(predictor.encoder.text) | set(predictor.encoder.numbers)
    expected_inputs = set(predictor.encoder.text) | ({NUMBERS_INPUT} if predictor.encoder.numbers else set())
    for network in predictor.networks:
        assert {tensor.name for tensor in network.inputs} == expected_inputs


def test_form_never_asks_for_later_grades_or_sex(predictor):
    form_columns = {field["column"] for field in predictor.form}
    assert not form_columns & {"G2", "G3", "sex"}


def test_typical_student_gets_three_probabilities_summing_to_one(predictor):
    probabilities = predictor.predict(predictor.typical)
    assert list(probabilities) == settings.CATEGORIES
    assert np.isclose(sum(probabilities.values()), 1, atol=1e-5)


def test_explanation_lists_only_changed_answers(predictor):
    assert predictor.explain(predictor.typical) == []
    field = next(field for field in predictor.form if field["column"] == "G1")
    student = {**predictor.typical, "G1": field["min"]}
    effects = predictor.explain(student)
    assert [effect["column"] for effect in effects] == ["G1"]
