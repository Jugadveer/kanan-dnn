"""The neural network, built with the Keras functional API, and a wrapper that
lets it be trained and scored like any scikit-learn model."""

from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from earlywarning import settings
from earlywarning.calibration import fit_temperature, scale_probabilities
from earlywarning.encoding import NUMBERS_INPUT, EmbeddingEncoder, OneHotInputs

VAL_BALANCED_ACCURACY = "val_balanced_accuracy"


@dataclass(frozen=True)
class Design:
    """The design choices the ablation study switches off one at a time."""
    embeddings: bool = True       # False: one-hot text columns instead
    focal: bool = True            # False: gamma = 0, plain (weighted) cross-entropy
    class_weights: bool = True    # False: every category weighs the same (alpha = 1)
    batch_norm: bool = True
    l2: bool = True


FULL_DESIGN = Design()


def class_alpha(y_train: np.ndarray) -> list[float]:
    """Per-category weight for focal loss: total / (categories x count).

    Needs support (the rarest) gets the largest weight, so the network cannot
    reach a low loss by mostly predicting On track.
    """
    counts = np.bincount(y_train, minlength=len(settings.CATEGORIES))
    return (len(y_train) / (len(counts) * counts)).round(3).tolist()


def build_network(vocabulary_sizes: dict[str, int], number_count: int, choices: dict,
                  alpha: list[float], design: Design = FULL_DESIGN) -> keras.Model:
    """Several inputs merged into one classifier.

    text column --> Embedding --\
    text column --> Embedding ---+--> Concatenate --> [Dense -> BatchNorm -> ReLU -> Dropout] x N --> softmax(3)
    numbers ----(standardised)--/

    choices holds the hyperparameters: embedding_size, units (one entry per
    hidden layer), dropout, l2 and learning_rate.
    """
    inputs, pieces = [], []

    for column, rows in vocabulary_sizes.items():
        code = keras.Input(shape=(), dtype="int32", name=column)
        # An embedding is a learnt lookup table: row k is the vector for value k.
        # A column with only 2 values never needs more than 2 dimensions.
        width = min(choices["embedding_size"], rows - 1)
        pieces.append(layers.Embedding(rows, width, name=f"{column}_embedding")(code))
        inputs.append(code)

    if number_count:
        numbers = keras.Input(shape=(number_count,), name=NUMBERS_INPUT)
        pieces.append(numbers)
        inputs.append(numbers)

    x = layers.Concatenate()(pieces) if len(pieces) > 1 else pieces[0]
    regulariser = keras.regularizers.L2(choices["l2"]) if design.l2 else None
    for units in choices["units"]:
        # With BatchNormalization right after, a bias is redundant: BN adds its own shift (beta).
        x = layers.Dense(units, use_bias=not design.batch_norm, kernel_regularizer=regulariser)(x)
        if design.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(choices["dropout"])(x)
    outputs = layers.Dense(len(settings.CATEGORIES), activation="softmax", name="category")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="early_warning")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=choices["learning_rate"]),
        # Focal loss = alpha x (1 - p)^gamma x cross-entropy. (1 - p)^gamma shrinks
        # the loss of students the model already gets right, so training focuses
        # on hard cases; alpha weights each category against its rarity.
        loss=keras.losses.CategoricalFocalCrossentropy(
            alpha=alpha if design.class_weights else 1.0,
            gamma=settings.CONFIG["training"]["focal_gamma"] if design.focal else 0.0),
        metrics=["accuracy"],
    )
    return model


def one_hot(y: np.ndarray) -> np.ndarray:
    """Category numbers -> one-hot rows, the label format focal loss expects."""
    return keras.utils.to_categorical(y, num_classes=len(settings.CATEGORIES))


def predict_proba(model: keras.Model, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Probability of each category for every row. training=False turns dropout
    off and makes BatchNormalization use its learnt averages."""
    return np.asarray(model(inputs, training=False))


class BalancedAccuracyLog(keras.callbacks.Callback):
    """Adds the validation balanced accuracy to each epoch's logs.

    Keras has no balanced-accuracy metric. Writing it into the logs lets
    KerasTuner rank models by it and shows it in the training history.
    """

    def __init__(self, val_inputs: dict[str, np.ndarray], y_val: np.ndarray):
        super().__init__()
        self.val_inputs = val_inputs
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        predicted = predict_proba(self.model, self.val_inputs).argmax(axis=1)
        logs[VAL_BALANCED_ACCURACY] = balanced_accuracy_score(self.y_val, predicted)


def train_network(model: keras.Model, train_inputs, y_train, val_inputs, y_val,
                  epochs: int, patience: int, extra_callbacks=(), initial_epoch: int = 0):
    """Fit with early stopping on validation loss, keeping the best epoch's weights."""
    callbacks = [
        BalancedAccuracyLog(val_inputs, y_val),  # first, so later callbacks can read it
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        *extra_callbacks,
    ]
    return model.fit(
        train_inputs, one_hot(y_train),
        validation_data=(val_inputs, one_hot(y_val)),
        epochs=epochs,
        initial_epoch=initial_epoch,
        batch_size=settings.CONFIG["training"]["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )


class NetworkClassifier:
    """Encoder + network(s) + temperature, trained and used like a scikit-learn
    model: fit(X, y) on raw student rows, then predict_proba(X).

    With several seeds it is a seed ensemble: one network per seed (different
    starting weights, dropout masks and batch order), probabilities averaged.
    The encoder, the focal-loss alpha and the weights all come from the rows
    given to fit(), and from nothing else.
    """

    def __init__(self, choices: dict, design: Design = FULL_DESIGN, seeds: tuple[int, ...] = (0,)):
        self.choices = choices
        self.design = design
        self.seeds = seeds
        self.temperature = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_stop: pd.DataFrame | None = None, y_stop: np.ndarray | None = None):
        """Train. Early stopping watches (X_stop, y_stop); without them, a
        stratified share of (X, y) is held back for it."""
        training = settings.CONFIG["training"]
        if X_stop is None:
            X, X_stop, y, y_stop = train_test_split(
                X, y, test_size=training["stopping_share"], stratify=y, random_state=settings.SEED)

        self.encoder = (EmbeddingEncoder() if self.design.embeddings else OneHotInputs()).fit(X)
        self.alpha = class_alpha(y)
        train_inputs, stop_inputs = self.encoder.transform(X), self.encoder.transform(X_stop)
        self.networks, self.histories = [], []
        for seed in self.seeds:
            keras.utils.set_random_seed(settings.SEED + seed)  # reproducible, different per member
            network = build_network(self.encoder.vocabulary_sizes(), self.encoder.number_count(),
                                    self.choices, self.alpha, self.design)
            self.histories.append(train_network(network, train_inputs, y, stop_inputs, y_stop,
                                                epochs=training["max_epochs"],
                                                patience=training["patience"]))
            self.networks.append(network)
        return self

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average of the networks' probabilities, before temperature scaling."""
        inputs = self.encoder.transform(X)
        return np.mean([predict_proba(network, inputs) for network in self.networks], axis=0)

    def calibrate(self, X_val: pd.DataFrame, y_val: np.ndarray) -> "NetworkClassifier":
        """Learn one temperature on validation students (see calibration.py)."""
        self.temperature = fit_temperature(self.raw_proba(X_val), y_val)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return scale_probabilities(self.raw_proba(X), self.temperature)
