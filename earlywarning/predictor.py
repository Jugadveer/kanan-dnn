"""Loads the trained network(s) and predicts one student, for the app."""

import json

import joblib
import keras
import numpy as np
import pandas as pd

from earlywarning import settings
from earlywarning.calibration import scale_probabilities
from earlywarning.explain import what_pushed
from earlywarning.model import predict_proba


def artifacts_exist() -> bool:
    return settings.SERVING_FILE.exists() and settings.ENCODER_FILE.exists() and settings.FORM_FILE.exists()


class RiskPredictor:
    def __init__(self):
        serving = json.loads(settings.SERVING_FILE.read_text())
        self.networks = [keras.models.load_model(settings.network_file(index), compile=False)
                         for index in range(serving["networks"])]
        self.temperature = serving["temperature"]
        self.encoder = joblib.load(settings.ENCODER_FILE)
        self.form = json.loads(settings.FORM_FILE.read_text())
        self.typical = {field["column"]: field["default"] for field in self.form}

    def probabilities(self, rows: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Average of the ensemble, then the temperature learnt on validation
        students (calibrated=False skips it, for the calibration comparison)."""
        inputs = self.encoder.transform(rows)
        averaged = np.mean([predict_proba(network, inputs) for network in self.networks], axis=0)
        return scale_probabilities(averaged, self.temperature) if calibrated else averaged

    def predict(self, student: dict) -> dict[str, float]:
        """Group -> probability, for one student given as {column: value}."""
        row = pd.DataFrame([student])
        return dict(zip(settings.CATEGORIES, self.probabilities(row)[0].tolist()))

    def explain(self, student: dict) -> list[dict]:
        """What pushed this student's prediction, largest effect first."""
        return what_pushed(self.probabilities, student, self.typical)
