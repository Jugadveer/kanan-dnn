"""Describes the app's input form, built from the columns the model kept."""

import pandas as pd

from earlywarning import settings


def describe_inputs(X_train: pd.DataFrame) -> list[dict]:
    """One entry per kept column, in the order of columns.yaml.

    Columns with "choices" list them as [value, words] pairs (JSON would turn
    number keys into strings, a list keeps 0 as 0). Plain counts get the
    smallest and largest value seen in training, so the app never asks about
    values the model has not seen. Defaults are the most common training value.
    """
    fields = []
    for column, info in settings.COLUMNS.items():
        if column not in X_train.columns:
            continue
        values = X_train[column]
        field = {"column": column, "group": info["group"], "question": info["question"],
                 "default": values.mode().tolist()[0]}
        if "choices" in info:
            seen = set(values.unique().tolist())
            field["choices"] = [[value, words] for value, words in info["choices"].items() if value in seen]
        else:
            field["min"], field["max"] = int(values.min()), int(values.max())
        fields.append(field)
    return fields
