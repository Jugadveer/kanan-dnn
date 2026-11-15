"""Reads config.yaml, columns.yaml and the app theme once, and exposes them
as plain constants for the rest of the package."""

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _read_yaml(name: str) -> dict:
    with open(ROOT / name, encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = _read_yaml("config.yaml")
COLUMNS = _read_yaml("columns.yaml")  # column -> {group, question, choices?}
GROUPS = ["Demographic", "Academic", "Social", "Behavioural"]  # the brief's four groups

# Paths
DATA_FILE = ROOT / CONFIG["data"]["folder"] / CONFIG["data"]["file"]
ARTIFACTS = ROOT / CONFIG["outputs"]["artifacts"]
REPORTS = ROOT / CONFIG["outputs"]["reports"]
CHARTS = REPORTS / "charts"

ENCODER_FILE = ARTIFACTS / "encoder.joblib"
SERVING_FILE = ARTIFACTS / "serving.json"  # temperature, typical student
FORM_FILE = ARTIFACTS / "form.json"
RESULTS_FILE = REPORTS / "results.json"
IMPORTANCE_FILE = REPORTS / "importance.json"
EXPERIMENTS_FILE = REPORTS / "experiments.json"


def network_file(index: int) -> Path:
    """One saved network per seed of the ensemble: network_0.keras, network_1.keras, ..."""
    return ARTIFACTS / f"network_{index}.keras"

# Inputs and target
LEFT_OUT = CONFIG["inputs"]["left_out"]
AUDITED_GROUPS = CONFIG["inputs"]["audited_groups"]
TARGET = CONFIG["target"]["column"]
CATEGORIES = [category["name"] for category in CONFIG["target"]["categories"]]
GRADE_CEILINGS = [category["max_grade"] for category in CONFIG["target"]["categories"]]

SEED = CONFIG["split"]["seed"]

# Names under which results are stored and shown.
NETWORK = "Neural network"
NAIVE = "Group of first-term grade"

# App colours, also used by the saved charts so they match the app.
THEME = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
