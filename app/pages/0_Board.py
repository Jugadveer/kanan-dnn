"""The board: the three groups, the headline results, and the watch list of
test students ranked by their chance of needing support."""

import pandas as pd
import streamlit as st

from earlywarning import settings
from earlywarning.evaluate import HEADLINES
from earlywarning.loading import inputs_and_target, read_students, three_way_split
from shared import group_lamp, grade_range, interval, load_json, load_predictor, risk_colour, lamp, start_page

NETWORK, NAIVE = settings.NETWORK, settings.NAIVE
SUPPORT = 0          # index of "Needs support" in settings.CATEGORIES
WATCH_ROWS = 15      # students shown on the watch list


@st.cache_data
def test_students() -> pd.DataFrame:
    """The held-out test students with the network's chances, highest risk first.
    The same split as training, so these students were never used to fit anything."""
    students = read_students()
    X, y = inputs_and_target(students)
    X_test, y_test = three_way_split(X, y)["test"]
    predictor = load_predictor()
    chances = predictor.probabilities(X_test[[field["column"] for field in predictor.form]])
    table = pd.DataFrame({
        "student": X_test.index,
        "G1": X_test["G1"].to_numpy(),
        "support_chance": chances[:, SUPPORT],
        "predicted": chances.argmax(axis=1),
        "actual": y_test,
        "G3": students.loc[X_test.index, settings.TARGET].to_numpy(),
    })
    return table.sort_values("support_chance", ascending=False, kind="stable")


def groups_table(results: dict) -> str:
    rows = "".join(
        f"<tr><td>{group_lamp(index)}{name}</td><td>{grade_range(index)}</td>"
        f"<td class='number'>{sum(counts[index] for counts in results['class_counts'].values())}</td></tr>"
        for index, name in enumerate(settings.CATEGORIES))
    return (f"<table class='watch'><tr><th>Group</th><th>Final grade (0–20)</th>"
            f"<th>Students in the data</th></tr>{rows}</table>")


def watch_list(table: pd.DataFrame) -> str:
    rows = []
    for rank, student in enumerate(table.head(WATCH_ROWS).itertuples(), start=1):
        flagged, needed = student.predicted == SUPPORT, student.actual == SUPPORT
        outcome = settings.CATEGORIES[student.actual]
        rows.append(
            f"<tr><td class='number'>{rank}</td><td class='number'>{student.student}</td>"
            f"<td class='number'>{student.G1}</td>"
            f"<td class='number'>{lamp(risk_colour(student.support_chance))}{student.support_chance:.0%}</td>"
            f"<td>{group_lamp(student.predicted)}{settings.CATEGORIES[student.predicted]}</td>"
            f"<td class='{'missed' if flagged != needed else ''}'>{group_lamp(student.actual)}"
            f"{outcome} ({student.G3})</td></tr>")
    return ("<table class='watch'><tr><th>Rank</th><th>Student</th><th>G1</th>"
            "<th>Chance of needing support</th><th>Predicted group</th><th>Actual final group (G3)</th></tr>"
            f"{''.join(rows)}</table>")


start_page("Board")
results = load_json(settings.RESULTS_FILE)
network, naive = results["test"][NETWORK], results["test"][NAIVE]
versus_naive = results["comparisons"][f"{NETWORK} vs {NAIVE}"]

st.write("After the first term a teacher knows one grade per student. This board estimates "
         "which of three groups each student's final Portuguese grade will fall into, while "
         "there are still two terms left to help.")

left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader("The three groups", anchor=False)
    st.html(groups_table(results))
with right:
    st.subheader(f"Results on {results['split_sizes']['test']} students the model never saw", anchor=False)
    st.table(pd.DataFrame({
        "Question": list(HEADLINES.values()),
        NETWORK: [interval(network, metric) for metric in HEADLINES],
        NAIVE: [interval(naive, metric) for metric in HEADLINES],
    }).set_index("Question"))
    mcnemar_p = versus_naive["test_mcnemar"]["p_value"]
    cv = versus_naive["cv_balanced_accuracy"]
    st.caption(
        "Brackets: 95 % bootstrap confidence intervals. The second column is the naive rule "
        "\"final group = group of the first-term grade\", the bar any model has to clear. "
        f"On these test students the difference in correct predictions is "
        f"{'significant' if mcnemar_p < 0.05 else 'not significant'} (McNemar p = {mcnemar_p:.2f}); "
        f"over 5 cross-validation folds the network's balanced accuracy is "
        f"{abs(cv['mean_difference']):.3f} {'higher' if cv['mean_difference'] > 0 else 'lower'} "
        f"on average, ahead in {cv['first_wins']} of 5 folds.")

table = test_students()
flagged = table["predicted"] == SUPPORT
needed = table["actual"] == SUPPORT
st.subheader("Watch list: test students, highest risk first", anchor=False)
st.write(f"The network flags **{flagged.sum()}** of {len(table)} test students as needing support. "
         f"**{(flagged & needed).sum()}** of them did finish below 10, and it caught "
         f"**{(flagged & needed).sum()} of the {needed.sum()}** students who did.")
st.html(watch_list(table))
st.caption(f"The {WATCH_ROWS} test students with the highest chance of needing support. "
           "Red text: the flag and the real outcome disagree. Student is the row number in the "
           "dataset; G3 is the real final grade.")
