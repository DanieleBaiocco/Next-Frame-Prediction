from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tensorflow as tf


def predict_autoregressive(
    model: tf.keras.Model,
    sequence: np.ndarray,
    input_frames: int = 3,
    num_predictions: int | None = None,
) -> np.ndarray:
    """Generate future frames recursively.

    sequence shape: (H, W, T), normalized to [0, 1].
    Returns a full timeline containing seed frames + predicted frames.
    """
    height, width, total_frames = sequence.shape

    if input_frames >= total_frames:
        raise ValueError("input_frames must be smaller than the sequence length.")

    if num_predictions is None:
        num_predictions = total_frames - input_frames

    if num_predictions < 1:
        raise ValueError("num_predictions must be >= 1.")

    generated = [
        sequence[:, :, i].astype(np.float32).copy()
        for i in range(input_frames)
    ]

    for _ in range(num_predictions):
        window = np.stack(generated[-input_frames:], axis=0)[..., np.newaxis]
        x = window[np.newaxis, ...]
        pred = model(x, training=False).numpy()[0, :, :, 0]
        generated.append(np.clip(pred, 0.0, 1.0))

    return np.stack(generated, axis=-1)


def _gray_to_rgb(frame: np.ndarray) -> np.ndarray:
    frame_u8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return np.repeat(frame_u8[..., np.newaxis], 3, axis=-1)


def save_prediction_video(
    predicted_timeline: np.ndarray,
    output_path: str | Path,
    fps: int = 5,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=None) as writer:
        for t in range(predicted_timeline.shape[-1]):
            writer.append_data(_gray_to_rgb(predicted_timeline[:, :, t]))

    return output_path


def save_comparison_video(
    ground_truth: np.ndarray,
    predicted_timeline: np.ndarray,
    output_path: str | Path,
    fps: int = 5,
) -> Path:
    """Save GT | prediction side by side for the overlapping timeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = min(ground_truth.shape[-1], predicted_timeline.shape[-1])
    h = ground_truth.shape[0]
    separator = np.zeros((h, 4, 3), dtype=np.uint8)

    with imageio.get_writer(output_path, fps=fps, codec="libx264", macro_block_size=None) as writer:
        for t in range(n_frames):
            gt = _gray_to_rgb(ground_truth[:, :, t])
            pred = _gray_to_rgb(predicted_timeline[:, :, t])
            frame = np.concatenate([gt, separator, pred], axis=1)
            writer.append_data(frame)

    return output_path
