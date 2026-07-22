from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_frames: int = 3, height: int = 64, width: int = 64) -> keras.Model:
    """ConvLSTM model: input frames -> next predicted frame."""
    inp = layers.Input(shape=(input_frames, height, width, 1), name="frames")

    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=False,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)

    out = layers.Conv2D(
        filters=1,
        kernel_size=(3, 3),
        activation="sigmoid",
        padding="same",
        name="next_frame",
    )(x)

    return keras.Model(inp, out, name="next_frame_convlstm")


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
    )
