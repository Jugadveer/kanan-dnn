"""Getting the data: download, read, label and split."""

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from earlywarning import settings


def fetch_csv() -> None:
    """Save student-por.csv from the UCI archive, unless it is already there.

    The archive nests a second zip (student.zip) that holds the CSV files.
    """
    if settings.DATA_FILE.exists():
        return
    url = settings.CONFIG["data"]["url"]
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    inner = zipfile.ZipFile(io.BytesIO(archive.read("student.zip")))
    settings.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    settings.DATA_FILE.write_bytes(inner.read(settings.DATA_FILE.name))


def read_students() -> pd.DataFrame:
    """All 649 students with all 33 columns. The file separates values with ';'."""
    fetch_csv()
    return pd.read_csv(settings.DATA_FILE, sep=";")


def category_of(grades: pd.Series) -> np.ndarray:
    """Final grade -> category number: 0 Needs support, 1 On track, 2 High achiever.

    np.searchsorted finds, for every grade, the first ceiling that is >= the
    grade: 7 -> ceiling 9 (index 0), 12 -> ceiling 13 (index 1).
    """
    return np.searchsorted(settings.GRADE_CEILINGS, grades.to_numpy())


def inputs_and_target(students: pd.DataFrame,
                      left_out: list[str] = settings.LEFT_OUT) -> tuple[pd.DataFrame, np.ndarray]:
    """The model's possible inputs (X) and the category (y).

    X keeps the students' row numbers as its index, so any row can be traced
    back to the full record (for the fairness audit and error analysis).
    """
    X = students.drop(columns=[settings.TARGET, *left_out])
    y = category_of(students[settings.TARGET])
    return X, y


def three_way_split(X: pd.DataFrame, y: np.ndarray) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Stratified 70 / 15 / 15 split into train, validation and test.

    Two cuts: first the test set off everything, then the validation set off
    the rest. Stratifying keeps the three categories in the same proportions.
    """
    split = settings.CONFIG["split"]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=split["test_share"], stratify=y, random_state=settings.SEED)
    validation_of_rest = split["validation_share"] / (1 - split["test_share"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=validation_of_rest, stratify=y_rest, random_state=settings.SEED)
    return {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}


def train_and_validation(parts: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Training and validation students together (551): the pool for
    cross-validated model selection. The test set is never part of it."""
    (X_train, y_train), (X_val, y_val) = parts["train"], parts["validation"]
    return pd.concat([X_train, X_val]), np.concatenate([y_train, y_val])
