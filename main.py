from __future__ import annotations

import argparse
from pathlib import Path
import sys


DEFAULT_OUTPUT = Path("outputs")
DEFAULT_DATA = Path("data/tfds")
INPUT_FRAMES = 3


def paths_from_output(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "root": root,
        "checkpoints": root / "checkpoints",
        "best_weights": root / "checkpoints" / "best.weights.h5",
        "best_metadata": root / "checkpoints" / "best.json",
        "history": root / "history.csv",
        "videos": root / "videos",
    }


def train(args: argparse.Namespace) -> None:
    import tensorflow as tf

    from src.checkpoints import PersistentBestCheckpoint
    from src.data import load_moving_mnist, make_fixed_dataset, make_training_dataset
    from src.model import build_model, compile_model

    paths = paths_from_output(args.output_dir)
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)

    print("[data] Loading Moving MNIST...")
    splits = load_moving_mnist(args.data_dir)

    train_ds = make_training_dataset(
        splits.train,
        batch_size=args.batch_size,
        input_frames=INPUT_FRAMES,
        seed=args.seed,
    )
    val_ds = make_fixed_dataset(
        splits.val,
        num_samples=args.val_samples,
        batch_size=args.batch_size,
        input_frames=INPUT_FRAMES,
        seed=args.seed + 1,
    )

    model = build_model(input_frames=INPUT_FRAMES)
    compile_model(model, learning_rate=args.learning_rate)

    if args.resume and paths["best_weights"].exists():
        model.load_weights(paths["best_weights"])
        print(f"[checkpoint] Resuming from: {paths['best_weights']}")
    else:
        print("[checkpoint] Starting from new weights.")

    checkpoint_cb = PersistentBestCheckpoint(
        paths["best_weights"],
        paths["best_metadata"],
        monitor="val_mse",
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mse",
        mode="min",
        patience=args.patience,
        restore_best_weights=False,
        verbose=1,
    )
    csv_logger_cb = tf.keras.callbacks.CSVLogger(
        str(paths["history"]),
        append=paths["history"].exists(),
    )

    print("[train] Starting training...")
    model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=val_ds,
        callbacks=[checkpoint_cb, early_stopping_cb, csv_logger_cb],
        verbose=1,
    )

    # Always leave the model in memory with the best persistent weights.
    if paths["best_weights"].exists():
        model.load_weights(paths["best_weights"])
        print(f"[train] Best weights loaded: {paths['best_weights']}")
    else:
        raise RuntimeError("Training ended without creating a best checkpoint.")


def generate(args: argparse.Namespace) -> None:
    from src.checkpoints import load_best_weights
    from src.data import get_sequence, load_moving_mnist
    from src.inference import (
        predict_autoregressive,
        save_comparison_video,
        save_prediction_video,
    )
    from src.model import build_model

    paths = paths_from_output(args.output_dir)

    print("[data] Loading Moving MNIST test split...")
    splits = load_moving_mnist(args.data_dir)
    sequence = get_sequence(splits.test, args.video_index)

    model = build_model(input_frames=INPUT_FRAMES)
    loaded = load_best_weights(model, paths["best_weights"])
    print(f"[checkpoint] Loaded best weights: {loaded}")

    max_gt_predictions = sequence.shape[-1] - INPUT_FRAMES
    num_predictions = args.frames if args.frames is not None else max_gt_predictions

    predicted = predict_autoregressive(
        model,
        sequence,
        input_frames=INPUT_FRAMES,
        num_predictions=num_predictions,
    )

    stem = f"video_{args.video_index:04d}"
    pred_path = paths["videos"] / f"{stem}_prediction.mp4"
    saved_pred = save_prediction_video(predicted, pred_path, fps=args.fps)

    print(f"[video] Prediction saved: {saved_pred}")

    if num_predictions <= max_gt_predictions:
        comparison_path = paths["videos"] / f"{stem}_gt_vs_prediction.mp4"
        saved_cmp = save_comparison_video(
            sequence,
            predicted,
            comparison_path,
            fps=args.fps,
        )
        print(f"[video] GT vs prediction saved: {saved_cmp}")
    else:
        print(
            "[video] Comparison video skipped because predictions extend "
            "beyond the available ground truth."
        )


def evaluate(args: argparse.Namespace) -> None:
    from src.checkpoints import load_best_weights
    from src.data import load_moving_mnist, make_fixed_dataset
    from src.model import build_model, compile_model

    paths = paths_from_output(args.output_dir)

    print("[data] Loading Moving MNIST...")
    splits = load_moving_mnist(args.data_dir)

    test_ds = make_fixed_dataset(
        splits.test,
        num_samples=args.test_samples,
        batch_size=args.batch_size,
        input_frames=INPUT_FRAMES,
        seed=args.seed,
    )

    model = build_model(input_frames=INPUT_FRAMES)
    compile_model(model, learning_rate=1e-3)
    loaded = load_best_weights(model, paths["best_weights"])
    print(f"[checkpoint] Loaded best weights: {loaded}")

    results = model.evaluate(test_ds, return_dict=True, verbose=1)
    print("\n[test]")
    for key, value in results.items():
        print(f"  {key}: {value:.8f}")


def _ask_int(label: str, default: int, minimum: int | None = None) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        return value


def _ask_float(label: str, default: float) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def interactive_menu() -> None:
    print(
        "\nNext Frame Prediction\n"
        "=====================\n"
        "1) Train / resume training\n"
        "2) Generate a video with the best model\n"
        "3) Evaluate the best model\n"
        "0) Exit\n"
    )
    choice = input("Choose an option: ").strip()

    if choice == "0":
        return

    output_dir = input(f"Output/checkpoint directory [{DEFAULT_OUTPUT}]: ").strip() or str(DEFAULT_OUTPUT)
    data_dir = input(f"TFDS data directory [{DEFAULT_DATA}]: ").strip() or str(DEFAULT_DATA)

    if choice == "1":
        epochs = _ask_int("Epochs", 30, 1)
        batch_size = _ask_int("Batch size", 32, 1)
        steps = _ask_int("Steps per epoch", 225, 1)
        val_samples = _ask_int("Validation samples", 960, 1)
        lr = _ask_float("Learning rate", 0.001)
        patience = _ask_int("Early-stopping patience", 8, 1)
        resume_raw = input("Resume from best checkpoint if present? [Y/n]: ").strip().lower()
        resume = resume_raw not in {"n", "no"}

        train(
            argparse.Namespace(
                output_dir=output_dir,
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps,
                val_samples=val_samples,
                learning_rate=lr,
                patience=patience,
                seed=42,
                resume=resume,
            )
        )
    elif choice == "2":
        video_index = _ask_int("Test video index (0-999)", 0, 0)
        fps = _ask_int("Video FPS", 5, 1)
        frames_raw = input("Future frames to generate [all available]: ").strip()
        frames = int(frames_raw) if frames_raw else None

        generate(
            argparse.Namespace(
                output_dir=output_dir,
                data_dir=data_dir,
                video_index=video_index,
                fps=fps,
                frames=frames,
            )
        )
    elif choice == "3":
        batch_size = _ask_int("Batch size", 32, 1)
        test_samples = _ask_int("Test samples", 960, 1)
        evaluate(
            argparse.Namespace(
                output_dir=output_dir,
                data_dir=data_dir,
                batch_size=batch_size,
                test_samples=test_samples,
                seed=999,
            )
        )
    else:
        print("Unknown option.")
        sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run inference for Moving-MNIST next-frame prediction."
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train the network and persist the best weights.")
    train_p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    train_p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--steps-per-epoch", type=int, default=225)
    train_p.add_argument("--val-samples", type=int, default=960)
    train_p.add_argument("--learning-rate", type=float, default=1e-3)
    train_p.add_argument("--patience", type=int, default=8)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore an existing best checkpoint and start from fresh weights.",
    )
    train_p.set_defaults(resume=True, func=train)

    gen_p = sub.add_parser(
        "generate",
        help="Generate an MP4 using the best saved checkpoint.",
    )
    gen_p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    gen_p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    gen_p.add_argument(
        "--video-index",
        type=int,
        default=0,
        help="Index in the 1000-sequence test split (0-999).",
    )
    gen_p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of future autoregressive frames. Default: all available GT frames.",
    )
    gen_p.add_argument("--fps", type=int, default=5)
    gen_p.set_defaults(func=generate)

    eval_p = sub.add_parser("evaluate", help="Evaluate the best checkpoint on test samples.")
    eval_p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    eval_p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    eval_p.add_argument("--batch-size", type=int, default=32)
    eval_p.add_argument("--test-samples", type=int, default=960)
    eval_p.add_argument("--seed", type=int, default=999)
    eval_p.set_defaults(func=evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        interactive_menu()
        return

    try:
        args.func(args)
    except (FileNotFoundError, IndexError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
