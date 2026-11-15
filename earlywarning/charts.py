"""Saved charts (PNG), drawn in the app's dark theme."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from earlywarning import settings

ACCENT = settings.THEME["primaryColor"]
PANEL = settings.THEME["secondaryBackgroundColor"]
INK = settings.THEME["textColor"]
GRID = settings.THEME["borderColor"]
MUTED = "#8a8f98"  # reference lines, error bars
# Charts compare up to three models: the network first (accent), then two muted colours.
MODEL_COLOURS = [ACCENT, "#8fa8c8", "#a3c49a"]
# Ordered categories get one hue from dark to light: Needs support -> High achiever.
CATEGORY_COLOURS = ["#8a6224", "#c08a35", "#f2cf8a"]

plt.rcParams.update({
    "figure.facecolor": settings.THEME["backgroundColor"],
    "axes.facecolor": settings.THEME["backgroundColor"],
    "savefig.facecolor": settings.THEME["backgroundColor"],
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": GRID, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig: plt.Figure, name: str) -> Path:
    path = settings.CHARTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")  # keep legends placed outside the axes
    plt.close(fig)
    return path


def rfecv_curve(scores_by_count: dict[int, list[float]], best: int, chosen: int) -> None:
    """Mean CV score (with +/- 1 std band) per column count; best and chosen marked."""
    counts = np.array(list(scores_by_count))
    means = np.array([mean for mean, _ in scores_by_count.values()])
    stds = np.array([std for _, std in scores_by_count.values()])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(counts, means, color=ACCENT)
    ax.fill_between(counts, means - stds, means + stds, color=ACCENT, alpha=0.15)
    ax.axvline(best, color=MUTED, linestyle=":", linewidth=1, label=f"best score: {best}")
    ax.axvline(chosen, color=INK, linestyle="--", linewidth=1, label=f"one-standard-error rule: {chosen}")
    ax.set(title="RFECV on the training split", xlabel="Encoded columns kept",
           ylabel="CV balanced accuracy")
    ax.legend(loc="lower right")
    save(fig, "rfecv_curve.png")


def training_history(history) -> None:
    """Focal loss per epoch, and the validation balanced accuracy."""
    fig, (loss_ax, score_ax) = plt.subplots(1, 2, figsize=(11, 3.8))
    loss_ax.plot(history.history["loss"], color=ACCENT, label="training")
    loss_ax.plot(history.history["val_loss"], color="#8fa8c8", label="validation")
    loss_ax.set(title="Focal loss", xlabel="Epoch")
    loss_ax.legend()
    score_ax.plot(history.history["val_balanced_accuracy"], color="#8fa8c8")
    score_ax.set(title="Validation balanced accuracy", xlabel="Epoch")
    save(fig, "training_history.png")


def confusion_matrices(y_test: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    colour_scale = LinearSegmentedColormap.from_list("panel_to_accent", [PANEL, ACCENT])
    fig, axes = plt.subplots(1, len(probabilities), figsize=(15, 4.6))
    for ax, (name, proba) in zip(axes, probabilities.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, proba.argmax(axis=1), display_labels=settings.CATEGORIES,
            cmap=colour_scale, colorbar=False, ax=ax)
        ax.set_title(name)
        ax.tick_params(axis="x", labelrotation=20)
    save(fig, "confusion_matrices.png")


def roc_curves(y_test: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    """One panel per category: that category against the other two."""
    fig, axes = plt.subplots(1, len(settings.CATEGORIES), figsize=(15, 4.4))
    for index, (ax, category) in enumerate(zip(axes, settings.CATEGORIES)):
        for position, (name, proba) in enumerate(probabilities.items()):
            RocCurveDisplay.from_predictions(
                y_test == index, proba[:, index], name=name, ax=ax,
                curve_kwargs={"color": MODEL_COLOURS[position]},
                plot_chance_level=(position == 0),
                chance_level_kw={"color": MUTED, "linestyle": ":"})
        ax.set(title=f"{category} vs the rest", xlabel="False positive rate",
               ylabel="True positive rate")
        ax.legend(loc="lower right", fontsize=8)
    save(fig, "roc_curves.png")


def importance_bars(rows: list[dict]) -> None:
    columns = [row["column"] for row in rows][::-1]
    drops = [row["mean_drop"] for row in rows][::-1]
    spread = [row["std_drop"] for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.2))
    ax.barh(columns, drops, xerr=spread, color=ACCENT, ecolor=MUTED)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set(title="Permutation importance (test set)", xlabel="Drop in balanced accuracy")
    save(fig, "permutation_importance.png")


def learning_curve(curve: dict[str, list[dict]]) -> None:
    """CV balanced accuracy (mean, with +/- 1 std band) against training-set size."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for colour, (name, points) in zip(MODEL_COLOURS, curve.items()):
        sizes = [point["students"] for point in points]
        means = np.array([point["mean"] for point in points])
        spreads = np.array([point["std"] for point in points])
        ax.plot(sizes, means, marker="o", color=colour, label=name)
        ax.fill_between(sizes, means - spreads, means + spreads, color=colour, alpha=0.15)
    ax.set(title="Learning curve (5-fold CV)", xlabel="Training students",
           ylabel="Balanced accuracy")
    ax.legend()
    save(fig, "learning_curve.png")


def reliability(before: list[dict], after: list[dict]) -> None:
    """Confidence of the predicted group against how often it is right.
    On the diagonal the probabilities can be taken at face value."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([1 / 3, 1], [1 / 3, 1], color=MUTED, linestyle=":", label="perfectly calibrated")
    for rows, colour, label in ((before, "#8fa8c8", "before scaling"), (after, ACCENT, "after scaling")):
        ax.plot([row["confidence"] for row in rows], [row["accuracy"] for row in rows],
                marker="o", color=colour, label=label)
    ax.set(title="Reliability diagram (test set)", xlabel="Average confidence",
           ylabel="Share predicted correctly", xlim=(0.3, 1.02), ylim=(0, 1.02))
    ax.legend(loc="lower right")
    save(fig, "reliability.png")


def ablation_bars(rows: list[dict]) -> None:
    """Change in CV balanced accuracy when one design choice is changed."""
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) + 1.2))
    differences = [row["difference"] for row in rows]
    ax.barh([row["variant"] for row in rows], differences,
            color=[ACCENT if value >= 0 else "#8fa8c8" for value in differences])
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set(title="Ablation: change against the full design (5-fold CV)",
           xlabel="Difference in balanced accuracy")
    save(fig, "ablation.png")
