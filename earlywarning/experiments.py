"""The longer analyses, run after training:

  repeated CV comparison, ablation study, learning curve, calibration,
  fairness audit, error analysis, and the cost of leaving sex out.

Run: python -m earlywarning experiments   (after python -m earlywarning train)

Nothing here changes the trained model. Analyses that use the test set only
describe the final model; every choice was made before, by cross-validation.
"""

import json
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline

from earlywarning import charts, settings
from earlywarning.baselines import TRADITIONAL, FirstTermGroup
from earlywarning.calibration import brier_score, expected_calibration_error, log_loss, reliability_bins
from earlywarning.encoding import flat_encoder
from earlywarning.evaluate import HEADLINES, NEEDS_SUPPORT, paired_folds, score
from earlywarning.loading import category_of, inputs_and_target, read_students, three_way_split, train_and_validation
from earlywarning.model import Design, NetworkClassifier
from earlywarning.predictor import RiskPredictor
from earlywarning.selection import select_columns
from earlywarning.validation import Fold, cross_validate, metric, plan_folds, summarise

NETWORK, NAIVE = settings.NETWORK, settings.NAIVE


class Context:
    """Data and decisions from the training run that every experiment needs."""

    def __init__(self):
        self.students = read_students()
        self.results = json.loads(settings.RESULTS_FILE.read_text())
        X, y = inputs_and_target(self.students)
        self.parts = three_way_split(X, y)
        self.X_pool, self.y_pool = train_and_validation(self.parts)
        chosen = self.results["model_selection"]
        self.network_settings = chosen["network"]["settings"]
        self.seeds = tuple(chosen["network"]["seeds"])
        self.best_traditional = chosen["best_traditional"]
        self.traditional_params = chosen["chosen_settings"][self.best_traditional]
        self.columns = self.results["selection"]["kept"]
        # The 5 folds of model selection, rebuilt from the stored column choices
        # (the split itself is deterministic), so no RFECV has to run again.
        splitter = StratifiedKFold(settings.CONFIG["model_selection"]["folds"], shuffle=True,
                                   random_state=settings.SEED)
        self.folds = [Fold(train_rows, held_out_rows, columns) for (train_rows, held_out_rows), columns
                      in zip(splitter.split(self.X_pool, self.y_pool), chosen["fold_columns"])]

    def network(self, design: Design = Design(), seeds=None) -> NetworkClassifier:
        return NetworkClassifier(self.network_settings, design, seeds or self.seeds)

    def traditional(self):
        return TRADITIONAL[self.best_traditional][0](self.traditional_params)

    def pool_without(self, left_out: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
        """The same 551 pool students with a different set of left-out columns."""
        X, y = inputs_and_target(self.students, left_out=left_out)
        return train_and_validation(three_way_split(X, y))


def repeated_comparison(context: Context) -> dict:
    """5-fold CV repeated 3 times with different shuffles (15 folds): the
    network against the naive rule and the best traditional model.

    Repeat 1 reuses the folds and scores from training; repeats 2 and 3 run
    RFECV again inside every new fold."""
    stored = context.results["model_selection"]["cv_folds"]
    scores = {NETWORK: list(stored[NETWORK]), NAIVE: list(stored[NAIVE]),
              context.best_traditional: list(stored[context.best_traditional])}
    for repeat in range(1, settings.CONFIG["model_selection"]["repeats"]):
        folds = plan_folds(context.X_pool, context.y_pool, repeat=repeat)
        scores[NETWORK] += cross_validate(context.network, context.X_pool, context.y_pool, folds)
        scores[NAIVE] += cross_validate(FirstTermGroup, context.X_pool, context.y_pool, folds)
        scores[context.best_traditional] += cross_validate(context.traditional, context.X_pool,
                                                           context.y_pool, folds)
    return {
        "folds": len(scores[NETWORK]),
        "summary": {name: summarise(fold_scores) for name, fold_scores in scores.items()},
        "network_vs": {rival: {name: paired_folds(metric(scores[NETWORK], name), metric(scores[rival], name))
                               for name in HEADLINES}
                       for rival in (NAIVE, context.best_traditional)},
    }


def ablation(context: Context) -> list[dict]:
    """Change one design choice at a time; 5-fold CV, one network per fold.

    The reference is the full design as a single network; the seed ensemble
    is its own row. Rows that change the inputs (selection rule, no selection,
    sex added back, no G1) build their own folds on the same students."""
    network_choice = context.results["model_selection"]["network"]
    reference = network_choice["single_network_folds"]
    variants = {
        f"Seed ensemble of {settings.CONFIG['model_selection']['ensemble_seeds']} networks":
            network_choice["ensemble_folds"],
        "One-hot inputs instead of embeddings": Design(embeddings=False),
        "Cross-entropy instead of focal loss (gamma = 0)": Design(focal=False),
        "No class weights (alpha = 1)": Design(class_weights=False),
        "No batch normalisation": Design(batch_norm=False),
        "No L2 regularisation": Design(l2=False),
    }
    rows = []
    for name, variant in variants.items():
        fold_scores = variant if isinstance(variant, list) else cross_validate(
            lambda variant=variant: context.network(variant, seeds=(0,)),
            context.X_pool, context.y_pool, context.folds)
        rows.append((name, fold_scores))

    single = partial(context.network, seeds=(0,))
    rows.append(("RFECV's best count instead of the one-standard-error rule", cross_validate(
        single, context.X_pool, context.y_pool, plan_folds(context.X_pool, context.y_pool, rule="best"))))
    rows.append(("No feature selection (all inputs)", cross_validate(
        single, context.X_pool, context.y_pool, plan_folds(context.X_pool, context.y_pool, select=False))))
    for name, left_out in (("Sex added back as an input", ["G2"]),
                           ("Without G1 (no grades at all)", ["G1", "G2", "sex"])):
        X, y = context.pool_without(left_out)
        rows.append((name, cross_validate(single, X, y, plan_folds(X, y))))

    return [{"variant": name,
             "balanced_accuracy": summarise(fold_scores)["balanced_accuracy"],
             "macro_f1": summarise(fold_scores)["macro_f1"]["mean"],
             "difference": float(np.mean(metric(fold_scores)) - np.mean(metric(reference))),
             **{key: value for key, value in paired_folds(metric(fold_scores), metric(reference)).items()
                if key in ("first_wins", "second_wins")}}
            for name, fold_scores in [("Full design (one network)", reference), *rows]]


def learning_curve(context: Context) -> dict[str, list[dict]]:
    """CV balanced accuracy when each fold trains on only part of its students.
    RFECV runs again on every reduced training set."""
    curve = {NETWORK: [], context.best_traditional: [], NAIVE: []}
    makers = {NETWORK: lambda: context.network(seeds=(0,)), context.best_traditional: context.traditional,
              NAIVE: FirstTermGroup}
    for share in settings.CONFIG["experiments"]["learning_curve_shares"]:
        folds = []
        for fold in context.folds:
            rows = fold.train_rows
            if share < 1:
                rows, _ = train_test_split(rows, train_size=share, stratify=context.y_pool[rows],
                                           random_state=settings.SEED)
            columns = fold.columns if share == 1 else select_columns(
                context.X_pool.iloc[rows], context.y_pool[rows])["kept"]
            folds.append(Fold(rows, fold.held_out_rows, columns))
        for name, make in makers.items():
            fold_scores = metric(cross_validate(make, context.X_pool, context.y_pool, folds))
            curve[name].append({"students": int(np.mean([len(fold.train_rows) for fold in folds])),
                                "mean": float(np.mean(fold_scores)), "std": float(np.std(fold_scores))})
    charts.learning_curve(curve)
    return curve


def test_predictions(context: Context, predictor: RiskPredictor):
    X_test, y_test = context.parts["test"]
    return X_test, y_test, predictor.probabilities(X_test[context.columns])


def calibration_report(context: Context, predictor: RiskPredictor) -> dict:
    """Test-set calibration of the final network before and after temperature scaling."""
    X_test, y_test = context.parts["test"]
    bins = settings.CONFIG["experiments"]["calibration_bins"]
    report = {"temperature": predictor.temperature}
    for label, calibrated in (("before", False), ("after", True)):
        probabilities = predictor.probabilities(X_test[context.columns], calibrated=calibrated)
        report[label] = {"ece": expected_calibration_error(probabilities, y_test, bins),
                         "brier": brier_score(probabilities, y_test),
                         "log_loss": log_loss(probabilities, y_test),
                         "bins": reliability_bins(probabilities, y_test, bins)}
    charts.reliability(report["before"]["bins"], report["after"]["bins"])
    return report


def fairness(context: Context, predictor: RiskPredictor) -> dict:
    """Test-set results per sex, school and home area, for the network and the
    naive rule; and how well the inputs can guess a student's sex (proxies)."""
    X_test, y_test, probabilities = test_predictions(context, predictor)
    predicted = probabilities.argmax(axis=1)
    naive = category_of(X_test["G1"])
    groups = {}
    for attribute in settings.AUDITED_GROUPS:
        values = context.students.loc[X_test.index, attribute].to_numpy()
        groups[attribute] = [{
            "value": value,
            "students": int(np.sum(values == value)),
            "actual_needs_support": int(np.sum((values == value) & (y_test == NEEDS_SUPPORT))),
            "flagged_needs_support": int(np.sum((values == value) & (predicted == NEEDS_SUPPORT))),
            "network_accuracy": float(np.mean(predicted[values == value] == y_test[values == value])),
            "naive_accuracy": float(np.mean(naive[values == value] == y_test[values == value])),
            "network_support_recall": _recall(y_test[values == value], predicted[values == value]),
        } for value in sorted(set(values))]

    # Proxy check: can the model's own inputs predict sex? (ROC-AUC 0.5 = no)
    sex = (context.students.loc[context.X_pool.index, "sex"] == "M").to_numpy()
    guesser = make_pipeline(flat_encoder(), LogisticRegression(max_iter=2000))
    guessed = cross_val_predict(guesser, context.X_pool[context.columns], sex, method="predict_proba",
                                cv=StratifiedKFold(5, shuffle=True, random_state=settings.SEED))[:, 1]
    return {"groups": groups, "sex_from_inputs_roc_auc": float(roc_auc_score(sex, guessed))}


def _recall(y_true: np.ndarray, predicted: np.ndarray) -> float | None:
    """Needs-support recall in a group, or None when the group has no such student."""
    needs_support = y_true == NEEDS_SUPPORT
    return float(np.mean(predicted[needs_support] == NEEDS_SUPPORT)) if needs_support.any() else None


def error_analysis(context: Context, predictor: RiskPredictor) -> dict:
    """The network's ten most confident mistakes on the test set, and patterns."""
    X_test, y_test, probabilities = test_predictions(context, predictor)
    predicted = probabilities.argmax(axis=1)
    wrong = predicted != y_test
    full = context.students.loc[X_test.index]
    first_term_group = category_of(X_test["G1"])
    worst = np.argsort(probabilities[np.arange(len(y_test)), y_test])[:10]
    return {
        "mistakes": int(wrong.sum()),
        "mistakes_matching_first_term_group": int(np.sum(wrong & (predicted == first_term_group))),
        "mean_grade_change_G1_to_G3_in_mistakes": float((full["G3"] - full["G1"])[wrong].mean()),
        "mean_grade_change_G1_to_G3_overall": float((full["G3"] - full["G1"]).mean()),
        "two_groups_off": int(np.sum(np.abs(predicted - y_test) == 2)),
        "worst": [{"G1": int(full["G1"].iloc[row]), "G3": int(full["G3"].iloc[row]),
                   "true": settings.CATEGORIES[y_test[row]], "predicted": settings.CATEGORIES[predicted[row]],
                   "probability_of_true_group": round(float(probabilities[row, y_test[row]]), 3),
                   "failures": int(full["failures"].iloc[row]), "absences": int(full["absences"].iloc[row])}
                  for row in worst],
    }


def cost_of_leaving_out_sex(context: Context) -> dict:
    """Train the final design with sex allowed (RFECV decides whether to use
    it) and score it on the test set next to the model without sex."""
    X, y = inputs_and_target(context.students, left_out=["G2"])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = three_way_split(X, y).values()
    columns = select_columns(X_train, y_train)["kept"]
    model = context.network().fit(X_train[columns], y_train, X_val[columns], y_val)
    model.calibrate(X_val[columns], y_val)
    probabilities = model.predict_proba(X_test[columns])
    flagged = probabilities.argmax(axis=1) == NEEDS_SUPPORT
    sex = context.students.loc[X_test.index, "sex"].to_numpy()
    return {"columns_kept": columns, "sex_used": "sex" in columns,
            "test": score(y_test, probabilities),
            "flagged_share_by_sex": {value: float(flagged[sex == value].mean()) for value in ("F", "M")}}


def run() -> None:
    started = time.perf_counter()
    context = Context()
    predictor = RiskPredictor()
    steps = {
        "repeated_cv": lambda: repeated_comparison(context),
        "ablation": lambda: ablation(context),
        "learning_curve": lambda: learning_curve(context),
        "calibration": lambda: calibration_report(context, predictor),
        "fairness": lambda: fairness(context, predictor),
        "errors": lambda: error_analysis(context, predictor),
        "sex_comparison": lambda: cost_of_leaving_out_sex(context),
    }
    output, minutes = {}, {}
    for name, step in steps.items():
        step_started = time.perf_counter()
        print(f"{name} ...")
        output[name] = step()
        minutes[name] = round((time.perf_counter() - step_started) / 60, 1)
        print(f"  done in {minutes[name]} min")
    charts.ablation_bars(output["ablation"][1:])
    output["minutes"] = {**minutes, "total": round((time.perf_counter() - started) / 60, 1)}
    settings.EXPERIMENTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"Saved {settings.EXPERIMENTS_FILE} in {output['minutes']['total']} minutes")
