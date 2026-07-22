# Next Frame Prediction — Moving MNIST

ConvLSTM network that receives **3 consecutive Moving-MNIST frames** and predicts the **next frame**.

This version is designed to work both:

- interactively from a terminal;
- through explicit CLI commands, useful for scripts, GitHub and Colab.

## 1. Install

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY>
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

A GPU is strongly recommended for training.

## 2. Interactive terminal menu

Run:

```bash
python main.py
```

You will get:

```text
Next Frame Prediction
=====================
1) Train / resume training
2) Generate a video with the best model
3) Evaluate the best model
0) Exit
```

This is the easiest entry point for someone who has just cloned the repository.

> **Important:** inference needs a trained `best.weights.h5`. After a fresh clone, either run training first or provide a pretrained checkpoint at `outputs/checkpoints/best.weights.h5`. If you want users to generate videos immediately after cloning, publish the checkpoint separately (for example as a GitHub Release asset or with Git LFS) and document where to place it.

## 3. Train

```bash
python main.py train
```

The best weights are saved automatically to:

```text
outputs/checkpoints/best.weights.h5
```

The checkpoint is selected using the lowest `val_mse`.

Metadata about the best checkpoint is saved to:

```text
outputs/checkpoints/best.json
```

Training history is appended to:

```text
outputs/history.csv
```

If `best.weights.h5` already exists, training resumes from it by default:

```bash
python main.py train
```

To deliberately start from fresh random weights:

```bash
python main.py train --no-resume
```

Example with custom parameters:

```bash
python main.py train \
  --epochs 50 \
  --batch-size 32 \
  --steps-per-epoch 225 \
  --patience 8
```

## 4. Generate a video

Generation always loads:

```text
outputs/checkpoints/best.weights.h5
```

Example:

```bash
python main.py generate --video-index 127
```

`--video-index` chooses one of the **1000 test sequences**, from `0` to `999`.

The command creates:

```text
outputs/videos/video_0127_prediction.mp4
outputs/videos/video_0127_gt_vs_prediction.mp4
```

In the comparison video:

```text
LEFT = ground truth
RIGHT = model prediction
```

The first 3 frames are the real seed frames. Subsequent frames are generated **autoregressively**: every prediction is reused as input for the next prediction.

Other examples:

```bash
python main.py generate --video-index 42 --fps 10
python main.py generate --video-index 42 --frames 30
```

If you generate beyond the 20 frames available in Moving MNIST, the prediction-only MP4 is still saved, but the GT-comparison video is skipped.

## 5. Evaluate the best model

```bash
python main.py evaluate
```

This evaluates the saved best checkpoint on samples taken from the real test split.

## 6. Use persistent storage, e.g. Google Drive

All persistent artifacts live under `--output-dir`.

Example:

```bash
python main.py train \
  --output-dir /content/drive/MyDrive/next_frame_prediction
```

Then inference must use the same directory:

```bash
python main.py generate \
  --video-index 127 \
  --output-dir /content/drive/MyDrive/next_frame_prediction
```

This means Colab can be disconnected without losing the best checkpoint.

## Important fixes compared with the old notebook

- Train / validation / test are now distinct:
  - train: sequences `0:8000`
  - validation: `8000:9000`
  - test: `9000:10000`
- The old generator yielded **inside the batch-building loop**, producing partially-filled batches. This is fixed.
- The old `test_dataset` was created from the validation generator. This is fixed.
- The metric is named correctly as MSE instead of “accuracy”.
- Best weights are persisted to disk, not only kept in RAM.
- Inference explicitly loads the best saved weights.
- Video generation is available from the terminal and supports choosing the test-video index.
