"""Every model's test scores with intervals, significance tests, and how the
network was chosen."""

import pandas as pd
import streamlit as st

from earlywarning import settings
from shared import chart_file, interval, load_json, start_page

start_page("Model report")
results = load_json(settings.RESULTS_FILE)
test = results["test"]
choice = results["model_selection"]
sizes = results["split_sizes"]
network = results["network"]


def score_table(metrics: dict[str, str]) -> pd.DataFrame:
    """Rows: models. Columns: metrics, each 'score (95 % interval)'."""
    return pd.DataFrame({label: [interval(result, key) for result in test.values()]
                         for key, label in metrics.items()}, index=list(test)).rename_axis("Model")


st.header("Model report", anchor=False)
st.write(
    f"Students were split {sizes['train']} / {sizes['validation']} / {sizes['test']} into "
    f"training, validation and test sets with the same mix of groups. Every model was chosen by "
    f"5-fold cross-validation on the {sizes['train'] + sizes['validation']} training and validation "
    f"students, with feature selection repeated inside each fold, then scored once on the "
    f"{sizes['test']} test students. Brackets are 95 % bootstrap intervals "
    f"({settings.CONFIG['evaluation']['bootstrap_resamples']} resamples)."
)

st.subheader("Needs support, yes or no", anchor=False)
st.table(score_table({"support_recall": "Recall", "support_precision": "Precision",
                      "support_roc_auc": "ROC-AUC", "support_accuracy": "Accuracy"}))
st.subheader("Which of the three groups", anchor=False)
st.table(score_table({"balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1",
                      "roc_auc": "ROC-AUC (one-vs-rest)", "within_one": "At most one group off"}))
st.caption("Balanced accuracy is the mean recall of the three groups, so ignoring the small "
           "Needs support group is punished; random guessing scores 0.33.")

st.subheader("Is the network really better?", anchor=False)
rows = []
for name, comparison in results["comparisons"].items():
    test_pair, folds = comparison["test_mcnemar"], comparison["cv_balanced_accuracy"]
    rows.append({
        "Comparison": name,
        "Test students only one model got right": f"{test_pair['only_first_right']} vs {test_pair['only_second_right']}",
        "McNemar p": f"{test_pair['p_value']:.3f}",
        "CV balanced accuracy difference": f"{folds['mean_difference']:+.3f}",
        "Folds won": f"{folds['first_wins']} of {folds['first_wins'] + folds['second_wins'] + folds['ties']}",
        "Paired t-test p": f"{folds['t_test_p_value']:.3f}",
    })
st.table(pd.DataFrame(rows).set_index("Comparison"))
st.caption("McNemar's test uses only the test students where exactly one model is right. "
           "The fold comparison uses the 5 cross-validation folds; its t-test treats the folds as "
           "independent, which they are not, so its p-value is optimistic. Below 0.05 counts as significant.")

st.subheader("How the network was chosen", anchor=False)
hyperband = results["hyperband"]
st.write(f"Hyperband trained {hyperband['trials']} candidate networks on the training split and "
         f"scored them on the 98 validation students. Its best "
         f"{len(choice['network']['candidates'])} distinct settings were then re-scored by "
         f"5-fold cross-validation on all {sizes['train'] + sizes['validation']} students:")
st.table(pd.DataFrame([{
    "Hidden layers": ", ".join(map(str, candidate["settings"]["units"])),
    "Embedding": str(candidate["settings"]["embedding_size"]),
    "Dropout": f"{candidate['settings']['dropout']:g}",
    "L2": f"{candidate['settings']['l2']:g}",
    "Learning rate": f"{candidate['settings']['learning_rate']:g}",
    "Hyperband (98 validation students)": f"{candidate['hyperband_validation']:.3f}",
    "Cross-validation (551 students)": f"{candidate['cv']['balanced_accuracy']['mean']:.3f} ± "
                                       f"{candidate['cv']['balanced_accuracy']['std']:.3f}",
} for candidate in choice["network"]["candidates"]]))
single = choice["network"]["single_network_cv"]["balanced_accuracy"]["mean"]
ensemble = choice["network"]["ensemble_cv"]["balanced_accuracy"]["mean"]
st.write(f"A seed ensemble ({settings.CONFIG['model_selection']['ensemble_seeds']} networks, "
         f"probabilities averaged) scored {ensemble:.3f} against {single:.3f} for one network, so it is "
         f"{'used' if choice['network']['ensemble_used'] else 'not used'}. Temperature scaling on the "
         f"validation students set T = {network['temperature']:.2f}.")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("The final network", anchor=False)
    st.markdown(
        f"- Inputs: {', '.join(results['selection']['kept'])} (chosen by RFECV)\n"
        f"- One embedding per text column (up to {network['embedding_size']} numbers), standardised numbers\n"
        f"- Hidden layers: {', '.join(map(str, network['units']))} units, each with batch "
        f"normalisation, ReLU and dropout {network['dropout']}\n"
        f"- L2 {network['l2']}, learning rate {network['learning_rate']}, focal loss with "
        f"alpha {network['focal_alpha']}\n"
        f"- {len(network['seeds'])} network(s) of {network['trainable_parameters_per_network']:,} "
        f"trainable weights"
    )
with right:
    st.subheader("Cross-validated scores", anchor=False)
    st.table(pd.DataFrame({
        "Balanced accuracy": [f"{cv['balanced_accuracy']['mean']:.3f} ± {cv['balanced_accuracy']['std']:.3f}"
                              for cv in choice["cv"].values()],
        "Macro F1": [f"{cv['macro_f1']['mean']:.3f}" for cv in choice["cv"].values()],
    }, index=list(choice["cv"])).rename_axis("Model"))
    st.caption("Traditional models' settings: " + "; ".join(
        f"{name}: {', '.join(f'{key} = {value}' for key, value in params.items())}"
        for name, params in choice["chosen_settings"].items() if params))

chart_file("confusion_matrices.png", "Rows: true group. Columns: predicted group.")
chart_file("roc_curves.png", "Each panel: one group against the other two. Curves closer to "
                             "the top-left corner are better; the dotted line is guessing.")
chart_file("training_history.png", "First network of the final model. Training stops once "
                                   "validation loss has not improved for 25 epochs, keeping the best epoch.")
