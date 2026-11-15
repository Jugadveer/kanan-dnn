"""Probability calibration: do the network's probabilities mean what they say?

A model is calibrated when, among students it gives 70 % to its top group,
about 70 % really are in that group.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def scale_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scaling: softmax(log p / T).

    log p are the network's logits up to a constant, so this is the usual
    "divide the logits by T". T > 1 softens over-confident probabilities,
    T < 1 sharpens under-confident ones; the predicted group never changes.
    """
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)  # avoid overflow in exp
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def log_loss(probabilities: np.ndarray, y: np.ndarray) -> float:
    """Average -log(probability of the true group)."""
    return float(-np.mean(np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(probabilities: np.ndarray, y: np.ndarray) -> float:
    """The temperature that minimises log loss on these (validation) students."""
    result = minimize_scalar(lambda t: log_loss(scale_probabilities(probabilities, t), y),
                             bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def brier_score(probabilities: np.ndarray, y: np.ndarray) -> float:
    """Mean squared distance between the probabilities and the one-hot truth,
    summed over the three groups: 0 is perfect, 2 is the worst possible."""
    truth = np.eye(probabilities.shape[1])[y]
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def reliability_bins(probabilities: np.ndarray, y: np.ndarray, bins: int) -> list[dict]:
    """Group students by the confidence of their predicted group (equal-width
    bins from 1/3 to 1) and compare average confidence with accuracy."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    edges = np.linspace(1 / probabilities.shape[1], 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence >= low) & (confidence <= high if high == 1.0 else confidence < high)
        if inside.any():
            rows.append({"low": float(low), "high": float(high), "students": int(inside.sum()),
                         "confidence": float(confidence[inside].mean()),
                         "accuracy": float(correct[inside].mean())})
    return rows


def expected_calibration_error(probabilities: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Weighted average gap between confidence and accuracy over the bins."""
    rows = reliability_bins(probabilities, y, bins)
    return float(sum(row["students"] * abs(row["confidence"] - row["accuracy"]) for row in rows) / len(y))
