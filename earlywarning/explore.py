"""Exploratory data analysis: data checks, class balance and charts.

Run: python -m earlywarning explore
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from earlywarning import charts, settings
from earlywarning.loading import category_of, inputs_and_target, read_students

# Columns whose category mix is drawn as stacked bars.
SPOTLIGHT = ["failures", "higher", "studytime", "Medu", "school", "Dalc"]


def check_data(students: pd.DataFrame) -> None:
    X, _ = inputs_and_target(students)
    print(f"Rows: {len(students)}   columns: {students.shape[1]}   "
          f"missing values: {students.isna().sum().sum()}   duplicate rows: {students.duplicated().sum()}")
    print(f"Inputs the model may use: {X.shape[1]} "
          f"(left out: {', '.join(settings.LEFT_OUT)}, and the target {settings.TARGET})")
    for group in settings.GROUPS:
        members = [column for column, info in settings.COLUMNS.items() if info["group"] == group]
        print(f"  {group} ({len(members)}): {', '.join(members)}")


def category_balance(students: pd.DataFrame) -> pd.Series:
    counts = pd.Series(category_of(students[settings.TARGET])).value_counts().sort_index()
    counts.index = settings.CATEGORIES
    print("\nStudents per category:")
    for name, count in counts.items():
        print(f"  {name:<15}{count:>4}  {count / len(students):.0%}")
    return counts


def first_term_versus_final(students: pd.DataFrame) -> None:
    """How far the first-term grade already tells the final group.

    Prints the table of G1 group against G3 group (rows: group of G1) and
    draws G1 against G3, with the group boundaries marked."""
    by_first_term = pd.crosstab(
        pd.Series(category_of(students["G1"]), name="group of G1").map(dict(enumerate(settings.CATEGORIES))),
        pd.Series(category_of(students[settings.TARGET]), name="final group").map(dict(enumerate(settings.CATEGORIES))),
    ).loc[settings.CATEGORIES, settings.CATEGORIES]
    same = sum(by_first_term.loc[name, name] for name in settings.CATEGORIES)
    print(f"\nGroup of G1 against final group (same group for {same} of {len(students)} "
          f"students, {same / len(students):.1%}):")
    print(by_first_term.to_string())
    print(f"Mean change from G1 to G3: {(students[settings.TARGET] - students['G1']).mean():+.2f} points")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    jitter = np.random.default_rng(settings.SEED).uniform(-0.25, 0.25, (2, len(students)))
    ax.scatter(students["G1"] + jitter[0], students[settings.TARGET] + jitter[1], s=10,
               color=charts.ACCENT, alpha=0.5)
    for ceiling in settings.GRADE_CEILINGS[:-1]:
        ax.axvline(ceiling + 0.5, color=charts.MUTED, linestyle=":")
        ax.axhline(ceiling + 0.5, color=charts.MUTED, linestyle=":")
    ax.set(title="First-term grade against final grade", xlabel="G1", ylabel="G3")
    charts.save(fig, "eda_first_term.png")


def chart_grades(students: pd.DataFrame, counts: pd.Series) -> None:
    """Histogram of G3 with each category's grade range shaded."""
    fig, (hist_ax, bar_ax) = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[2, 1])
    lower = 0
    for ceiling, colour in zip(settings.GRADE_CEILINGS, charts.CATEGORY_COLOURS):
        hist_ax.axvspan(lower - 0.5, ceiling + 0.5, color=colour, alpha=0.15)
        lower = ceiling + 1
    hist_ax.hist(students[settings.TARGET], bins=range(0, 22), align="left",
                 color=charts.ACCENT, rwidth=0.85)
    hist_ax.set(title="Final grade G3 (shading = category ranges)", xlabel="Grade (0-20)",
                ylabel="Students")
    bar_ax.bar(counts.index, counts.values, color=charts.CATEGORY_COLOURS)
    bar_ax.set(title="Students per category")
    charts.save(fig, "eda_grades.png")


def chart_rank_correlation(students: pd.DataFrame) -> None:
    """Spearman correlation of every numeric input with G3.

    Spearman compares ranks rather than raw values, so it suits ordered codes
    such as studytime (1-4) and does not assume a straight-line relationship.
    """
    X, _ = inputs_and_target(students)
    numeric = X.select_dtypes("number")
    rho = numeric.corrwith(students[settings.TARGET], method="spearman").sort_values()
    print("\nSpearman correlation with G3:")
    print(rho.round(2).to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = [charts.ACCENT if value > 0 else "#8fa8c8" for value in rho]
    ax.barh(rho.index, rho.values, color=colours)
    ax.axvline(0, color=charts.GRID, linewidth=1)
    ax.set(title="Spearman correlation with the final grade", xlabel="rho")
    charts.save(fig, "eda_rank_correlation.png")


def chart_category_mix(students: pd.DataFrame) -> None:
    """For a few columns: the share of each category among students with each value."""
    categories = pd.Series(category_of(students[settings.TARGET])).map(dict(enumerate(settings.CATEGORIES)))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for column, ax in zip(SPOTLIGHT, axes.flat):
        shares = pd.crosstab(students[column], categories, normalize="index")[settings.CATEGORIES]
        shares.plot.bar(stacked=True, ax=ax, color=charts.CATEGORY_COLOURS, width=0.75,
                        legend=False, rot=0, edgecolor=settings.THEME["backgroundColor"])
        ax.set(title=settings.COLUMNS[column]["question"], xlabel="", ylim=(0, 1))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.04))
    charts.save(fig, "eda_category_mix.png")


def run() -> None:
    students = read_students()
    check_data(students)
    counts = category_balance(students)
    first_term_versus_final(students)
    chart_grades(students, counts)
    chart_rank_correlation(students)
    chart_category_mix(students)
    print(f"\nCharts saved in {settings.CHARTS}")
