"""The main training run: tune, choose by cross-validation, train, test once, save.

Run: python -m earlywarning train
"""

import json
import time

import joblib
import keras
import numpy as np
import tensorflow as tf

from earlywarning import charts, settings
from earlywarning.baselines import TRADITIONAL, FirstTermGroup, search_space, trivial_models
from earlywarning.encoding import EmbeddingEncoder
from earlywarning.evaluate import bootstrap_intervals, mcnemar, paired_folds, per_category, score
from earlywarning.explain import permutation_importance
from earlywarning.form import describe_inputs
from earlywarning.loading import inputs_and_target, read_students, three_way_split, train_and_validation
from earlywarning.model import NetworkClassifier, class_alpha
from earlywarning.selection import select_columns
from earlywarning.tuning import EarlyWarningHyperModel, best_trials, distinct_candidates, run_hyperband
from earlywarning.validation import cross_validate, metric, plan_folds, summarise

NETWORK, NAIVE = settings.NETWORK, settings.NAIVE


def class_counts(y) -> list[int]:
    return np.bincount(y, minlength=len(settings.CATEGORIES)).tolist()


def uses_selection(model) -> bool:
    """False only for the naive rule, which always reads G1."""
    return getattr(model, "uses_selected_columns", True)


def propose_with_hyperband(X_train, y_train, X_val, y_val, columns) -> tuple[list[dict], dict]:
    """Hyperband on the 70 % training split, scored on the 15 % validation split.
    Returns its best distinct settings, to be re-scored by cross-validation."""
    encoder = EmbeddingEncoder().fit(X_train[columns])
    hypermodel = EarlyWarningHyperModel(encoder.vocabulary_sizes(), encoder.number_count(),
                                        class_alpha(y_train))
    started = time.perf_counter()
    tuner = run_hyperband(hypermodel, encoder.transform(X_train[columns]), y_train,
                          encoder.transform(X_val[columns]), y_val)
    summary = {"trials": len(tuner.oracle.trials),
               "minutes": round((time.perf_counter() - started) / 60, 1),
               "best_trials": best_trials(tuner)}
    count = settings.CONFIG["model_selection"]["network_candidates"]
    return distinct_candidates(tuner, count), summary


def choose_network(candidates: list[dict], X_pool, y_pool, folds) -> dict:
    """Re-score Hyperband's candidates with 5-fold CV on the 551 pool students,
    pick the best, then decide by the same CV whether a seed ensemble helps.

    Hyperband judged each candidate on only 98 validation students, so its
    winner is partly luck; the CV mean uses all 551 students."""
    for candidate in candidates:
        candidate["cv"] = cross_validate(lambda: NetworkClassifier(candidate["settings"]),
                                         X_pool, y_pool, folds)
        print(f"  {candidate['settings']}: Hyperband validation {candidate['val_balanced_accuracy']:.3f}, "
              f"CV {np.mean(metric(candidate['cv'])):.3f}")
    best = max(candidates, key=lambda candidate: np.mean(metric(candidate["cv"])))

    seeds = tuple(range(settings.CONFIG["model_selection"]["ensemble_seeds"]))
    ensemble_cv = cross_validate(lambda: NetworkClassifier(best["settings"], seeds=seeds),
                                 X_pool, y_pool, folds)
    use_ensemble = np.mean(metric(ensemble_cv)) > np.mean(metric(best["cv"]))
    print(f"  seed ensemble of {len(seeds)}: CV {np.mean(metric(ensemble_cv)):.3f} -> "
          f"{'used' if use_ensemble else 'not used'}")
    return {"settings": best["settings"], "seeds": seeds if use_ensemble else (0,),
            "cv": ensemble_cv if use_ensemble else best["cv"],
            "candidates": [{"settings": c["settings"], "hyperband_validation": c["val_balanced_accuracy"],
                            "cv": summarise(c["cv"])} for c in candidates],
            "single_network_cv": summarise(best["cv"]), "ensemble_cv": summarise(ensemble_cv),
            "single_network_folds": best["cv"], "ensemble_folds": ensemble_cv,
            "ensemble_used": bool(use_ensemble)}


def choose_baselines(X_pool, y_pool, folds) -> dict[str, dict]:
    """Best settings of each traditional model by the same CV, plus the naive
    rule and the trivial models (nothing to tune)."""
    chosen = {}
    for family, (build, _) in TRADITIONAL.items():
        scored = [(params, cross_validate(lambda: build(params), X_pool, y_pool, folds))
                  for params in search_space(family)]
        params, fold_scores = max(scored, key=lambda pair: np.mean(metric(pair[1])))
        chosen[family] = {"params": params, "build": lambda build=build, params=params: build(params),
                          "cv": fold_scores}
        print(f"  {family}: {params}, CV {np.mean(metric(fold_scores)):.3f} "
              f"({len(scored)} settings tried)")
    chosen[NAIVE] = {"params": {}, "build": FirstTermGroup,
                     "cv": cross_validate(FirstTermGroup, X_pool, y_pool, folds)}
    for name, model in trivial_models().items():
        chosen[name] = {"params": {}, "build": lambda model=model: model,
                        "cv": cross_validate(lambda model=model: model, X_pool, y_pool, folds)}
    return chosen


def fit_final_models(network_choice, others, X_train, y_train, X_val, y_val, columns) -> dict:
    """Every chosen model, refitted on the training split. The network stops
    early on the validation split, then learns its temperature there."""
    network = NetworkClassifier(network_choice["settings"], seeds=network_choice["seeds"])
    network.fit(X_train[columns], y_train, X_val[columns], y_val)
    network.calibrate(X_val[columns], y_val)
    print(f"\nTemperature from validation: {network.temperature:.3f}")
    charts.training_history(network.histories[0])
    finals = {NETWORK: network}
    for name, choice in others.items():
        model = choice["build"]()
        finals[name] = model.fit(X_train[columns] if uses_selection(model) else X_train, y_train)
    return finals


def test_once(finals, network_choice, others, best_traditional, X_test, y_test, columns) -> tuple[dict, dict]:
    """Scores with 95 % intervals for every model, and the network compared with
    the naive rule and the best traditional model (McNemar on test, paired CV folds)."""
    probabilities = {name: model.predict_proba(X_test[columns] if uses_selection(model) else X_test)
                     for name, model in finals.items()}
    test = {name: {"scores": score(y_test, proba), "intervals": bootstrap_intervals(y_test, proba)}
            for name, proba in probabilities.items()}
    predicted = {name: proba.argmax(axis=1) for name, proba in probabilities.items()}
    comparisons = {
        f"{NETWORK} vs {rival}": {
            "test_mcnemar": mcnemar(y_test, predicted[NETWORK], predicted[rival]),
            "cv_balanced_accuracy": paired_folds(metric(network_choice["cv"]), metric(others[rival]["cv"])),
            "cv_macro_f1": paired_folds(metric(network_choice["cv"], "macro_f1"),
                                        metric(others[rival]["cv"], "macro_f1")),
        }
        for rival in (NAIVE, best_traditional)
    }
    for name, proba in probabilities.items():
        print(f"\n=== {name} (test set) ===\n{per_category(y_test, proba)}")
    shown = {name: probabilities[name] for name in (NETWORK, NAIVE, best_traditional)}
    charts.confusion_matrices(y_test, shown)
    charts.roc_curves(y_test, shown)
    return test, comparisons


def save_app_files(network, network_choice, X_train_columns) -> None:
    """What the app loads: the network(s), the encoder, the temperature and the form."""
    for index, member in enumerate(network.networks):
        member.save(settings.network_file(index))
    joblib.dump(network.encoder, settings.ENCODER_FILE)
    settings.SERVING_FILE.write_text(json.dumps({
        "networks": len(network.networks), "temperature": network.temperature,
        "settings": network_choice["settings"]}, indent=2))
    settings.FORM_FILE.write_text(json.dumps(describe_inputs(X_train_columns), indent=2))


def run() -> None:
    started = time.perf_counter()
    for folder in (settings.ARTIFACTS, settings.CHARTS):
        folder.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(settings.SEED)
    tf.config.experimental.enable_op_determinism()  # same numbers on every run

    # 1. Data: what is known after the first term, split 70/15/15.
    X, y = inputs_and_target(read_students())
    parts = three_way_split(X, y)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = parts.values()
    X_pool, y_pool = train_and_validation(parts)
    for name, (part_X, part_y) in parts.items():
        print(f"{name:<11}{len(part_X):>4} students, per group {class_counts(part_y)}")

    # 2. Feature selection on the training split (used by Hyperband and the final models).
    selection = select_columns(X_train, y_train)
    columns = selection["kept"]
    charts.rfecv_curve(selection["scores_by_count"], selection["best_count"], selection["encoded_kept"])
    print(f"\nRFECV on the training split kept {columns}")

    # 3. Hyperband proposes, cross-validation decides.
    print("\nHyperband search ...")
    candidates, hyperband = propose_with_hyperband(X_train, y_train, X_val, y_val, columns)
    print("\nCross-validating on train + validation (RFECV inside every fold) ...")
    folds = plan_folds(X_pool, y_pool)
    network_choice = choose_network(candidates, X_pool, y_pool, folds)
    others = choose_baselines(X_pool, y_pool, folds)
    best_traditional = max(TRADITIONAL, key=lambda family: np.mean(metric(others[family]["cv"])))

    # 4. Final models, then 5. the test set, used once.
    finals = fit_final_models(network_choice, others, X_train, y_train, X_val, y_val, columns)
    test, comparisons = test_once(finals, network_choice, others, best_traditional, X_test, y_test, columns)
    network = finals[NETWORK]

    # 6. What the network relies on (analysis only: nothing is chosen from it).
    importance = permutation_importance(network.predict_proba, X_test[columns], y_test)
    charts.importance_bars(importance)

    # 7. Save what the app loads, and the results.
    save_app_files(network, network_choice, X_train[columns])
    settings.IMPORTANCE_FILE.write_text(json.dumps(importance, indent=2))
    settings.RESULTS_FILE.write_text(json.dumps({
        "split_sizes": {name: len(part_X) for name, (part_X, _) in parts.items()},
        "class_counts": {name: class_counts(part_y) for name, (_, part_y) in parts.items()},
        "selection": selection,
        "hyperband": hyperband,
        "network": {**network_choice["settings"], "focal_alpha": network.alpha,
                    "seeds": list(network_choice["seeds"]), "temperature": round(network.temperature, 4),
                    "trainable_parameters_per_network": sum(
                        int(np.prod(w.shape)) for w in network.networks[0].trainable_weights),
                    "epochs_trained": [len(history.epoch) for history in network.histories]},
        "model_selection": {
            "network": {key: value for key, value in network_choice.items() if key != "cv"},
            "fold_columns": [fold.columns for fold in folds],
            "cv": {NETWORK: summarise(network_choice["cv"]),
                   **{name: summarise(choice["cv"]) for name, choice in others.items()}},
            # every metric on every fold, reused by the experiments for paired comparisons
            "cv_folds": {NETWORK: network_choice["cv"],
                         **{name: choice["cv"] for name, choice in others.items()}},
            "chosen_settings": {name: choice["params"] for name, choice in others.items()},
            "best_traditional": best_traditional,
        },
        "test": test,
        "comparisons": comparisons,
        "total_minutes": round((time.perf_counter() - started) / 60, 1),
    }, indent=2))

    for name, result in test.items():
        s, ci = result["scores"], result["intervals"]
        print(f"{name:<30} balanced acc {s['balanced_accuracy']:.3f} "
              f"[{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}]  "
              f"macro F1 {s['macro_f1']:.3f}  support recall {s['support_recall']:.3f}")
    for name, comparison in comparisons.items():
        print(f"{name}: McNemar p = {comparison['test_mcnemar']['p_value']:.3f}, "
              f"CV mean difference {comparison['cv_balanced_accuracy']['mean_difference']:+.3f}")
    print(f"\nDone in {(time.perf_counter() - started) / 60:.1f} minutes.")
