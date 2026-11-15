"""Results of `python -m earlywarning experiments`: repeated CV, ablation,
learning curve, calibration, fairness, error analysis and the cost of leaving sex out."""

import pandas as pd
import streamlit as st

from earlywarning import settings
from earlywarning.evaluate import HEADLINES
from shared import chart_file, load_json, start_page


start_page("Experiments")
st.header("Experiments", anchor=False)
if not settings.EXPERIMENTS_FILE.exists():
    st.info("These analyses come from `python -m earlywarning experiments`, which has not been run yet.")
    st.stop()
experiments = load_json(settings.EXPERIMENTS_FILE)

# Repeated cross-validation
repeated = experiments["repeated_cv"]
st.subheader(f"Repeated cross-validation ({repeated['folds']} folds)", anchor=False)
st.write("5-fold cross-validation repeated 3 times with different shuffles, on the 551 training and "
         "validation students, feature selection inside every fold. More reliable than one test set "
         "of 98 students.")
st.table(pd.DataFrame({
    HEADLINES[metric]: [f"{summary[metric]['mean']:.3f} ± {summary[metric]['std']:.3f}"
                           for summary in repeated["summary"].values()]
    for metric in HEADLINES}, index=list(repeated["summary"])).rename_axis("Model"))
st.table(pd.DataFrame([{
    "Network against": rival, "Metric": HEADLINES[metric],
    "Mean difference": f"{result['mean_difference']:+.3f}",
    "Folds network ahead / behind / tied": f"{result['first_wins']} / {result['second_wins']} / {result['ties']}",
    "Paired t-test p (optimistic)": f"{result['t_test_p_value']:.3f}",
} for rival, by_metric in repeated["network_vs"].items() for metric, result in by_metric.items()]).set_index("Network against"))

# Ablation
st.subheader("Ablation: one change at a time", anchor=False)
st.table(pd.DataFrame([{
    "Change": row["variant"],
    "Balanced accuracy (5-fold CV)": f"{row['balanced_accuracy']['mean']:.3f} ± {row['balanced_accuracy']['std']:.3f}",
    "Difference": f"{row['difference']:+.3f}",
    "Folds better / worse": f"{row['first_wins']} / {row['second_wins']}",
} for row in experiments["ablation"]]).set_index("Change"))
chart_file("ablation.png", "Each bar: the change in cross-validated balanced accuracy against the full design.")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Learning curve", anchor=False)
    chart_file("learning_curve.png", "Balanced accuracy as each fold trains on 20 % to 100 % of its "
                                     "students (feature selection redone each time).")
with right:
    st.subheader("Calibration", anchor=False)
    calibration = experiments["calibration"]
    st.table(pd.DataFrame({
        "Expected calibration error": [f"{calibration[stage]['ece']:.3f}" for stage in ("before", "after")],
        "Brier score": [f"{calibration[stage]['brier']:.3f}" for stage in ("before", "after")],
        "Log loss": [f"{calibration[stage]['log_loss']:.3f}" for stage in ("before", "after")],
    }, index=["Before temperature scaling", f"After (T = {calibration['temperature']:.2f})"]))
    chart_file("reliability.png", "Test students grouped by confidence. On the dotted line the "
                                  "probabilities mean exactly what they say.")

# Fairness
st.subheader("Fairness audit (test set)", anchor=False)
fairness = experiments["fairness"]
labels = {"sex": "Sex", "school": "School", "address": "Home area"}
st.table(pd.DataFrame([{
    "Group": f"{labels[attribute]}: {row['value']}",
    "Students": row["students"],
    "Needs support (actual)": row["actual_needs_support"],
    "Flagged by the network": row["flagged_needs_support"],
    "Network accuracy": f"{row['network_accuracy']:.2f}",
    "Naive rule accuracy": f"{row['naive_accuracy']:.2f}",
    "Needs-support recall": "–" if row["network_support_recall"] is None else f"{row['network_support_recall']:.2f}",
} for attribute, rows in fairness["groups"].items() for row in rows]).set_index("Group"))
st.caption(f"Sex is not an input. How well the model's inputs can guess a student's sex "
           f"(ROC-AUC, 0.5 = not at all): {fairness['sex_from_inputs_roc_auc']:.2f}. "
           f"Groups of a few dozen students: differences of 0.1 can be chance.")

sex = experiments["sex_comparison"]
st.write(f"**With sex allowed as an input** (RFECV {'kept' if sex['sex_used'] else 'did not keep'} it), "
         f"the same design scores {sex['test']['balanced_accuracy']:.3f} balanced accuracy and "
         f"{sex['test']['support_recall']:.2f} needs-support recall on the test set, and flags "
         f"{sex['flagged_share_by_sex']['F']:.0%} of girls and {sex['flagged_share_by_sex']['M']:.0%} of boys.")

# Errors
st.subheader("The network's worst mistakes", anchor=False)
errors = experiments["errors"]
st.write(f"{errors['mistakes']} test students in the wrong group; in "
         f"{errors['mistakes_matching_first_term_group']} of them the network predicted the group of "
         f"the first-term grade. {errors['two_groups_off']} mistakes are two groups off. Average change "
         f"from G1 to G3: {errors['mean_grade_change_G1_to_G3_in_mistakes']:+.1f} points in the mistakes, "
         f"{errors['mean_grade_change_G1_to_G3_overall']:+.1f} overall.")
st.dataframe(pd.DataFrame(errors["worst"]), hide_index=True, width="stretch")

st.caption(f"Run time of the experiments: {experiments['minutes']['total']} minutes.")
