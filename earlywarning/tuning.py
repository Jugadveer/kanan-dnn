"""Hyperparameter search with KerasTuner's Hyperband."""

import keras
import keras_tuner as kt

from earlywarning import settings
from earlywarning.model import VAL_BALANCED_ACCURACY, build_network, train_network


class EarlyWarningHyperModel(kt.HyperModel):
    """Tells KerasTuner what to vary (build) and how to train a candidate (fit)."""

    def __init__(self, vocabulary_sizes: dict[str, int], number_count: int, alpha: list[float]):
        super().__init__()
        self.vocabulary_sizes = vocabulary_sizes
        self.number_count = number_count
        self.alpha = alpha

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        space = settings.CONFIG["tuning"]
        layer_count = hp.Int("hidden_layers", 1, space["max_hidden_layers"])
        choices = {
            "embedding_size": hp.Choice("embedding_size", space["embedding_sizes"]),
            "units": [hp.Choice(f"units_{layer}", space["units"]) for layer in range(layer_count)],
            "dropout": hp.Float("dropout", **space["dropout"]),
            "l2": hp.Choice("l2", space["l2_strengths"]),
            "learning_rate": hp.Choice("learning_rate", space["learning_rates"]),
        }
        return build_network(self.vocabulary_sizes, self.number_count, choices, self.alpha)

    def fit(self, hp, model, x, y, validation_data, epochs, callbacks=(), initial_epoch=0, **_):
        """Hyperband decides the epochs; its own callbacks (which save the best
        epoch) run after ours, so they can read the balanced accuracy we log."""
        val_inputs, y_val = validation_data
        return train_network(model, x, y, val_inputs, y_val, epochs=epochs,
                             patience=settings.CONFIG["tuning"]["patience"],
                             extra_callbacks=callbacks, initial_epoch=initial_epoch)


def run_hyperband(hypermodel: EarlyWarningHyperModel, train_inputs, y_train,
                  val_inputs, y_val) -> kt.Hyperband:
    """Hyperband: start many models on a small epoch budget, keep the best third,
    give them three times the epochs, and repeat up to max_epochs. Weak settings
    are dropped early, so more of them fit into the same time."""
    space = settings.CONFIG["tuning"]
    tuner = kt.Hyperband(
        hypermodel,
        objective=kt.Objective(VAL_BALANCED_ACCURACY, direction="max"),
        max_epochs=space["max_epochs"],
        factor=space["reduction_factor"],
        seed=settings.SEED,
        directory=settings.REPORTS / "tuner",
        project_name="hyperband",
        overwrite=True,
    )
    tuner.search(train_inputs, y_train, validation_data=(val_inputs, y_val), verbose=0)
    return tuner


def chosen_settings(hp: kt.HyperParameters) -> dict:
    """The settings a trial actually used, without Hyperband's bookkeeping
    (tuner/epochs, tuner/round, ...) and without units of unused layers."""
    values = hp.values
    return {
        "hidden_layers": values["hidden_layers"],
        "units": [values[f"units_{layer}"] for layer in range(values["hidden_layers"])],
        "embedding_size": values["embedding_size"],
        "dropout": round(values["dropout"], 2),
        "l2": values["l2"],
        "learning_rate": values["learning_rate"],
    }


def best_trials(tuner: kt.Hyperband, count: int = 10) -> list[dict]:
    """The top trials: their settings, epoch budget and validation balanced accuracy."""
    return [
        {**chosen_settings(trial.hyperparameters),
         "epochs": trial.hyperparameters.values["tuner/epochs"],
         VAL_BALANCED_ACCURACY: round(trial.score, 4)}
        for trial in tuner.oracle.get_best_trials(count)
    ]


def distinct_candidates(tuner: kt.Hyperband, count: int) -> list[dict]:
    """Hyperband's best `count` different settings, best first.

    The same settings can appear several times (once per Hyperband round), so
    repeats are skipped. Each candidate keeps its validation score, which the
    cross-validation re-scoring is later compared with.
    """
    candidates = []
    for row in best_trials(tuner, count=len(tuner.oracle.trials)):
        settings_only = {key: value for key, value in row.items()
                         if key not in ("epochs", VAL_BALANCED_ACCURACY)}
        if all(settings_only != candidate["settings"] for candidate in candidates):
            candidates.append({"settings": settings_only, VAL_BALANCED_ACCURACY: row[VAL_BALANCED_ACCURACY]})
        if len(candidates) == count:
            break
    return candidates
