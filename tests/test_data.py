"""The dataset and the target: what the brief and config.yaml promise."""

import numpy as np
import pandas as pd

from earlywarning import settings
from earlywarning.loading import category_of, inputs_and_target, read_students, three_way_split


def test_dataset_has_649_students_and_33_columns():
    students = read_students()
    assert students.shape == (649, 33)
    assert students.isna().sum().sum() == 0


def test_inputs_keep_first_term_grade_and_drop_later_grades_and_sex():
    X, _ = inputs_and_target(read_students())
    assert "G1" in X.columns
    for column in ("G2", "G3", "sex"):
        assert column not in X.columns
    assert X.shape[1] == 30  # 33 columns minus G2, G3 and sex


def test_sex_is_left_out_by_default_in_config():
    assert "sex" in settings.LEFT_OUT and "G2" in settings.LEFT_OUT


def test_group_edges():
    grades = pd.Series([0, 9, 10, 13, 14, 20])
    assert category_of(grades).tolist() == [0, 0, 1, 1, 2, 2]


def test_split_sizes_are_disjoint_and_stratified():
    X, y = inputs_and_target(read_students())
    parts = three_way_split(X, y)
    sizes = {name: len(part_X) for name, (part_X, _) in parts.items()}
    assert sizes == {"train": 453, "validation": 98, "test": 98}
    indexes = [set(part_X.index) for part_X, _ in parts.values()]
    assert not (indexes[0] & indexes[1] or indexes[0] & indexes[2] or indexes[1] & indexes[2])
    for _, part_y in parts.values():  # every part keeps roughly 15 / 55 / 30 %
        assert np.allclose(np.bincount(part_y) / len(part_y), np.bincount(y) / len(y), atol=0.02)
