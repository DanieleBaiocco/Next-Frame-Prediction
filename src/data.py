from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass(frozen=True)
class DatasetSplits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def load_moving_mnist(data_dir: str | Path | None = None) -> DatasetSplits:
    """Load Moving MNIST as uint8 arrays shaped (N, H, W, T).

    Keeping uint8 avoids converting the whole dataset to float32 in RAM.
    Batches are normalized to [0, 1] only when they are sampled.
    """
    ds = tfds.as_numpy(
        tfds.load(
            "moving_mnist",
            split="test",
            batch_size=-1,
            data_dir=str(data_dir) if data_dir else None,
        )
    )
    sequences = np.asarray(ds["image_sequence"])

    # TFDS normally returns (N, T, H, W, 1).
    if sequences.ndim == 5 and sequences.shape[-1] == 1:
        sequences = np.squeeze(sequences, axis=-1)

    if sequences.ndim != 4:
        raise ValueError(
            f"Unexpected Moving MNIST shape: {sequences.shape}. "
            "Expected (N,T,H,W,1) or (N,T,H,W)."
        )

    # Convert (N, T, H, W) -> (N, H, W, T).
    sequences = np.transpose(sequences, (0, 2, 3, 1))

    if len(sequences) < 10_000:
        raise ValueError(f"Expected at least 10000 sequences, got {len(sequences)}.")

    return DatasetSplits(
        train=sequences[:8000],
        val=sequences[8000:9000],
        test=sequences[9000:10000],
    )


def _sample_batch(
    dataset: np.ndarray,
    batch_size: int,
    input_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a complete batch for next-frame prediction."""
    n_sequences, height, width, sequence_len = dataset.shape
    window_len = input_frames + 1
    if sequence_len < window_len:
        raise ValueError(
            f"Sequence length {sequence_len} is too short for {input_frames} input frames."
        )

    indices = rng.integers(0, n_sequences, size=batch_size)
    starts = rng.integers(0, sequence_len - window_len + 1, size=batch_size)

    x = np.empty((batch_size, input_frames, height, width, 1), dtype=np.float32)
    y = np.empty((batch_size, height, width, 1), dtype=np.float32)

    for i, (idx, start) in enumerate(zip(indices, starts)):
        window = dataset[idx, :, :, start : start + window_len].astype(np.float32) / 255.0
        x[i, ..., 0] = np.transpose(window[:, :, :input_frames], (2, 0, 1))
        y[i, ..., 0] = window[:, :, input_frames]

    return x, y


def batch_generator(
    dataset: np.ndarray,
    batch_size: int = 32,
    input_frames: int = 3,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    while True:
        yield _sample_batch(dataset, batch_size, input_frames, rng)


def make_training_dataset(
    dataset: np.ndarray,
    batch_size: int = 32,
    input_frames: int = 3,
    seed: int = 42,
) -> tf.data.Dataset:
    _, height, width, _ = dataset.shape

    output_signature = (
        tf.TensorSpec(
            shape=(batch_size, input_frames, height, width, 1),
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(batch_size, height, width, 1),
            dtype=tf.float32,
        ),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: batch_generator(dataset, batch_size, input_frames, seed),
        output_signature=output_signature,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def make_fixed_dataset(
    dataset: np.ndarray,
    num_samples: int,
    batch_size: int = 32,
    input_frames: int = 3,
    seed: int = 123,
) -> tf.data.Dataset:
    """Build a deterministic validation/test set sampled once."""
    rng = np.random.default_rng(seed)
    x, y = _sample_batch(dataset, num_samples, input_frames, rng)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_sequence(dataset: np.ndarray, video_index: int) -> np.ndarray:
    if not 0 <= video_index < len(dataset):
        raise IndexError(
            f"video_index must be between 0 and {len(dataset) - 1}, got {video_index}."
        )
    return dataset[video_index].astype(np.float32) / 255.0
