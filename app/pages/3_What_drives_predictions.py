"""Which inputs the network relies on, and which feature selection removed."""

import altair as alt
import pandas as pd
import streamlit as st

from earlywarning import settings
from shared import ACCENT, chart_file, load_json, start_page

start_page("What drives predictions")
results = load_json(settings.RESULTS_FILE)
importance = pd.DataFrame(load_json(settings.IMPORTANCE_FILE))
importance["question"] = importance["column"].map(lambda column: settings.COLUMNS[column]["question"])

st.header("What drives predictions", anchor=False)
st.write(
    "Permutation importance of the final network: shuffle one detail across the test students, so it no "
    "longer belongs to the right person, and measure how much balanced accuracy falls. "
    f"Each detail was shuffled {settings.CONFIG['explain']['shuffles']} times; the bar is "
    "the average drop. Near zero or below means the network barely uses it."
)

bars = alt.Chart(importance).mark_bar(color=ACCENT).encode(
    y=alt.Y("question:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
    x=alt.X("mean_drop:Q", title="Drop in balanced accuracy"),
    tooltip=[alt.Tooltip("question:N", title="Detail"),
             alt.Tooltip("mean_drop:Q", title="Average drop", format=".3f"),
             alt.Tooltip("std_drop:Q", title="Spread (std)", format=".3f")],
)
st.altair_chart(
    bars.properties(
        height=28 * len(importance),
        padding={"left": 70},  # room for the long question labels, so none are clipped
    ),
    width="stretch",
)

selection = results["selection"]
st.subheader("Left out by feature selection", anchor=False)
def count(number: int, word: str) -> str:
    """'1 detail', '2 details'."""
    return f"{number} {word}{'' if number == 1 else 's'}"


st.write(
    f"RFECV started from all {len(selection['kept']) + len(selection['dropped'])} allowed details "
    f"({selection['encoded_total']} columns after one-hot encoding). Its best score came with "
    f"{count(selection['best_count'], 'encoded column')}; the one-standard-error rule kept "
    f"{count(selection['encoded_kept'], 'encoded column')}, from "
    f"{count(len(selection['kept']), 'detail')}: {', '.join(selection['kept'])}. "
    f"The form does not ask for the others. Sex and the second-term grade G2 are never offered to the model."
)
fold_columns = results["model_selection"]["fold_columns"]
kept_counts = pd.Series([column for columns in fold_columns for column in columns]).value_counts()
st.table(pd.DataFrame({
    "Detail": [settings.COLUMNS[column]["question"] for column in kept_counts.index],
    f"Kept in how many of the {len(fold_columns)} cross-validation folds": kept_counts.values,
}).set_index("Detail"))
st.caption("RFECV runs again inside every cross-validation fold. A detail kept in every fold "
           "is a stable choice; one kept in a single fold is borderline.")
chart_file("rfecv_curve.png", "Cross-validated balanced accuracy for each number of encoded columns.")

st.subheader("In the data", anchor=False)
chart_file("eda_first_term.png", "First-term grade against final grade. Dotted lines: group "
                                 "boundaries. Points off the diagonal blocks changed group.")
chart_file("eda_category_mix.png", "Share of each group among students with each answer.")
chart_file("eda_rank_correlation.png", "Spearman rank correlation of each numeric detail with the final grade.")
