"""Helpers used by more than one page of the app."""

import json
from pathlib import Path

import streamlit as st

from earlywarning import settings
from earlywarning.predictor import RiskPredictor, artifacts_exist

ACCENT = st.get_option("theme.primaryColor")
TITLE = "Early warning after the first term"
STYLE_FILE = settings.ROOT / ".streamlit" / "board.css"

# One lamp colour per group, in the order of settings.CATEGORIES
# (Needs support, On track, High achiever). Same values as board.css.
GROUP_COLOURS = ["#e5484d", "#8fa8c8", "#3fb67b"]
# The risk meter runs green -> amber -> red along its length.
RISK_COLOURS = [(0.25, "#3fb67b"), (0.50, "#e0a43a"), (1.00, "#e5484d")]
METER_CELLS = 20


def start_page(title: str) -> None:
    """Browser-tab title, and a clear stop if nothing is trained yet."""
    st.set_page_config(page_title=f"{title} · {TITLE}", layout="wide")
    if not artifacts_exist() or not settings.RESULTS_FILE.exists():
        st.error("The model has not been trained on this computer yet, so there is nothing to show.")
        st.markdown("From the project folder, run:\n\n"
                    "```\npython -m earlywarning train\n```\n\n"
                    "It takes about 15 minutes. Then reload this page.")
        st.stop()


@st.cache_resource  # load the networks once per server, not on every click
def load_predictor() -> RiskPredictor:
    return RiskPredictor()


@st.cache_data
def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def grade_range(category_index: int) -> str:
    """'0–9' for Needs support, '10–13' for On track, '14–20' for High achiever."""
    low = 0 if category_index == 0 else settings.GRADE_CEILINGS[category_index - 1] + 1
    return f"{low}–{settings.GRADE_CEILINGS[category_index]}"


def interval(result: dict, metric: str) -> str:
    """'0.812 (0.74–0.88)': a test score with its 95 % bootstrap interval."""
    low, high = result["intervals"][metric]
    return f"{result['scores'][metric]:.3f} ({low:.2f}–{high:.2f})"


def chart_file(name: str, caption: str) -> None:
    """Show a chart saved by the training or experiments run."""
    path = settings.CHARTS / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Chart {name} is made by `python -m earlywarning experiments`; it has not been run yet.")


# ---------------------------------------------------------------- board pieces (HTML)
def lamp(colour: str) -> str:
    return f"<span class='lamp' style='background:{colour}'></span>"


def group_lamp(category_index: int) -> str:
    return lamp(GROUP_COLOURS[category_index])


def risk_colour(position: float) -> str:
    """Green below 25 %, amber below 50 %, red from 50 % up."""
    return next(colour for limit, colour in RISK_COLOURS if position < limit or limit == 1.0)


def meter(share: float, colour_of_cell) -> str:
    """A bar of METER_CELLS cells; cells up to `share` are lit. `colour_of_cell`
    gives a lit cell's colour from the cell's position (0 to 1) along the bar."""
    lit = round(share * METER_CELLS)
    cells = [f"<span style='background:{colour_of_cell(index / METER_CELLS)}'></span>"
             if index < lit else "<span></span>" for index in range(METER_CELLS)]
    return f"<div class='meter'>{''.join(cells)}</div>"


def board_header() -> None:
    """The board's styling and the strip across the top of every page."""
    st.html(f"<style>{STYLE_FILE.read_text()}</style>")
    st.html("""
        <div class="board-header">
          <div class="name">Early warning <span class="term">· after term 1</span></div>
          <div class="note">Which group a student's final Portuguese grade will fall in,
            known after the first term · UCI Student Performance, 649 students</div>
        </div>""")
