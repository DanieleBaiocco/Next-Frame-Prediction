from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf


class PersistentBestCheckpoint(tf.keras.callbacks.Callback):
    """Save weights only when val_mse improves across current AND previous runs."""

    def __init__(
        self,
        weights_path: str | Path,
        metadata_path: str | Path,
        monitor: str = "val_mse",
    ) -> None:
        super().__init__()
        self.weights_path = Path(weights_path)
        self.metadata_path = Path(metadata_path)
        self.monitor = monitor
        self.best = float("inf")

        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self.metadata_path.exists():
            try:
                metadata = json.loads(self.metadata_path.read_text())
                self.best = float(metadata.get("best_val_mse", float("inf")))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        current = float(current)
        if current < self.best:
            previous = self.best
            self.best = current
            self.model.save_weights(self.weights_path)

            metadata = {
                "best_val_mse": current,
                "epoch": int(epoch) + 1,
                "monitor": self.monitor,
            }
            self.metadata_path.write_text(json.dumps(metadata, indent=2))

            previous_text = "none" if previous == float("inf") else f"{previous:.8f}"
            print(
                f"\n[checkpoint] New best {self.monitor}: {current:.8f} "
                f"(previous: {previous_text})"
            )
            print(f"[checkpoint] Saved: {self.weights_path}")


def load_best_weights(model: tf.keras.Model, weights_path: str | Path) -> Path:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {path}\n"
            "Run training first with `python main.py train`."
        )
    model.load_weights(path)
    return path
