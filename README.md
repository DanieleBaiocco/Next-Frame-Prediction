# Next Frame Prediction with ConvLSTM

A small **video prediction** project based on **Moving MNIST** and a stacked **ConvLSTM** network.

The model receives **3 consecutive 64×64 grayscale frames** and predicts the next frame. During inference, predictions are generated **autoregressively**: each predicted frame is fed back into the model to generate the following one.

The dataset is loaded through **TensorFlow Datasets** and is automatically downloaded if it is not already available locally.

## Prediction examples

Each example compares the original Moving MNIST sequence with the model output:

**Left: ground truth — Right: autoregressive prediction**

<table>
  <tr>
    <td><img src="outputs/videos/video_0002_gt_vs_prediction.gif" width="300" alt="Prediction example 2"></td>
    <td><img src="outputs/videos/video_0003_gt_vs_prediction.gif" width="300" alt="Prediction example 3"></td>
    <td><img src="outputs/videos/video_0127_gt_vs_prediction.gif" width="300" alt="Prediction example 127"></td>
  </tr>
</table>

## Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd next_frame_prediction
pip install -r requirements.txt
```

A GPU is recommended for training, but inference can also run on CPU.

## Command-line usage

Running the program without arguments opens an interactive menu:

```bash
python main.py
```

The same operations can also be executed directly from the command line.

### Generate a prediction video

```bash
python main.py generate --video-index 127
```

The command automatically loads the best available checkpoint from:

```text
outputs/checkpoints/best.weights.h5
```

Generated videos are saved under `outputs/videos/`.

Example with custom length and playback speed:

```bash
python main.py generate --video-index 127 --frames 17 --fps 20
```

Main options:

| Option | Description | Default |
|---|---|---:|
| `--video-index` | Test sequence to use (`0`–`999`) | `0` |
| `--frames` | Number of future frames generated autoregressively | all available GT frames |
| `--fps` | Output video playback speed | `5` |
| `--output-dir` | Base directory containing checkpoints and generated videos | `outputs` |
| `--data-dir` | TensorFlow Datasets cache directory | `data/tfds` |

Two files are normally produced:

```text
outputs/videos/video_0127_prediction.mp4
outputs/videos/video_0127_gt_vs_prediction.mp4
```

Moving MNIST sequences contain 20 frames. Since the first 3 are used as input, up to **17 future frames** have matching ground truth. If `--frames` is greater than 17, the model can still generate a longer prediction video, but the ground-truth comparison video is skipped.

### Train / resume training

```bash
python main.py train
```

Training monitors validation MSE and automatically saves the best weights to:

```text
outputs/checkpoints/best.weights.h5
```

If that checkpoint already exists, training resumes from those weights by default. Early stopping is enabled and stops training after several epochs without validation improvement.

Useful options:

```bash
python main.py train \
  --epochs 30 \
  --batch-size 32 \
  --steps-per-epoch 225 \
  --learning-rate 0.001 \
  --patience 8
```

| Option | Description | Default |
|---|---|---:|
| `--epochs` | Maximum number of epochs | `30` |
| `--batch-size` | Training batch size | `32` |
| `--steps-per-epoch` | Training steps per epoch | `225` |
| `--val-samples` | Number of validation samples | `960` |
| `--learning-rate` | Adam learning rate | `0.001` |
| `--patience` | Early-stopping patience | `8` |
| `--no-resume` | Start from fresh weights instead of loading the best checkpoint | disabled |

### Evaluate the best model

```bash
python main.py evaluate
```

This loads `best.weights.h5` and evaluates it on the test split.

Optional parameters include `--batch-size`, `--test-samples`, `--seed`, `--output-dir`, and `--data-dir`.

For the complete CLI reference:

```bash
python main.py --help
python main.py train --help
python main.py generate --help
python main.py evaluate --help
```

## Model overview

The network is composed of three stacked **ConvLSTM2D** layers with batch normalization, followed by a **Conv2D** output layer. It is trained with **mean squared error (MSE)** to predict the next frame.

Dataset split used by the project:

- 8,000 sequences for training
- 1,000 sequences for validation
- 1,000 sequences for testing
