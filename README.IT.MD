# Next Frame Prediction con ConvLSTM

Un piccolo progetto di **predizione video** basato su **Moving MNIST** e su una rete **ConvLSTM** impilata.

Il modello riceve **3 frame consecutivi in scala di grigi da 64×64 pixel** e predice il frame successivo. Durante l'inferenza, le predizioni vengono generate in modo **autoregressivo**: ogni frame predetto viene reinserito nel modello per generare quello successivo.

Il dataset viene caricato tramite **TensorFlow Datasets** e viene scaricato automaticamente se non è già disponibile in locale.

## Esempi di predizione

Ogni esempio confronta la sequenza originale di Moving MNIST con l'output del modello:

**Sinistra: ground truth — Destra: predizione autoregressiva**

<table>
  <tr>
    <td><img src="outputs/videos/video_0002_gt_vs_prediction.gif" width="300" alt="Esempio di predizione 2"></td>
    <td><img src="outputs/videos/video_0003_gt_vs_prediction.gif" width="300" alt="Esempio di predizione 3"></td>
    <td><img src="outputs/videos/video_0127_gt_vs_prediction.gif" width="300" alt="Esempio di predizione 127"></td>
  </tr>
</table>

## Installazione

```bash
git clone <REPOSITORY_URL>
cd Next-Frame-Prediction

py -3.12 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Per il training è consigliata una GPU, mentre l'inferenza può essere eseguita anche su CPU.

## Utilizzo da linea di comando

Avviando il programma senza argomenti si apre un menu interattivo:

```bash
python main.py
```

Le stesse operazioni possono essere eseguite direttamente da linea di comando.

### Generare un video di predizione

```bash
python main.py generate --video-index 127
```

Il comando carica automaticamente il miglior checkpoint disponibile da:

```text
outputs/checkpoints/best.weights.h5
```

I video generati vengono salvati in `outputs/videos/`.

Esempio con lunghezza e velocità di riproduzione personalizzate:

```bash
python main.py generate --video-index 127 --frames 17 --fps 20
```

Opzioni principali:

| Opzione | Descrizione | Default |
|---|---|---:|
| `--video-index` | Sequenza del test set da utilizzare (`0`–`999`) | `0` |
| `--frames` | Numero di frame futuri generati autoregressivamente | tutti i frame GT disponibili |
| `--fps` | Velocità di riproduzione del video generato | `5` |
| `--output-dir` | Directory base contenente checkpoint e video generati | `outputs` |
| `--data-dir` | Directory di cache di TensorFlow Datasets | `data/tfds` |

Normalmente vengono prodotti due file:

```text
outputs/videos/video_0127_prediction.mp4
outputs/videos/video_0127_gt_vs_prediction.mp4
```

Le sequenze Moving MNIST contengono 20 frame. Poiché i primi 3 vengono usati come input, sono disponibili fino a **17 frame futuri** con una ground truth corrispondente. Se `--frames` è maggiore di 17, il modello può comunque generare un video di predizione più lungo, ma il video di confronto con la ground truth viene saltato.

### Training / ripresa del training

```bash
python main.py train
```

Durante il training viene monitorata la MSE di validazione e i pesi migliori vengono salvati automaticamente in:

```text
outputs/checkpoints/best.weights.h5
```

Se il checkpoint esiste già, per impostazione predefinita il training riparte da quei pesi. È inoltre attivo l'**early stopping**, che interrompe automaticamente il training dopo diverse epoche senza miglioramenti sulla validation.

Esempio con alcuni parametri personalizzati:

```bash
python main.py train \
  --epochs 30 \
  --batch-size 32 \
  --steps-per-epoch 225 \
  --learning-rate 0.001 \
  --patience 8
```

| Opzione | Descrizione | Default |
|---|---|---:|
| `--epochs` | Numero massimo di epoche | `30` |
| `--batch-size` | Dimensione del batch di training | `32` |
| `--steps-per-epoch` | Numero di step di training per epoca | `225` |
| `--val-samples` | Numero di campioni di validazione | `960` |
| `--learning-rate` | Learning rate dell'ottimizzatore Adam | `0.001` |
| `--patience` | Patience dell'early stopping | `8` |
| `--no-resume` | Avvia il training da pesi casuali invece di caricare il miglior checkpoint | disabilitato |

### Valutare il miglior modello

```bash
python main.py evaluate
```

Questo comando carica `best.weights.h5` e valuta il modello sul test set.

Tra i parametri opzionali sono disponibili `--batch-size`, `--test-samples`, `--seed`, `--output-dir` e `--data-dir`.

Per vedere tutti i comandi e le opzioni disponibili:

```bash
python main.py --help
python main.py train --help
python main.py generate --help
python main.py evaluate --help
```

## Panoramica del modello

La rete è composta da tre layer **ConvLSTM2D** impilati con batch normalization, seguiti da un layer di output **Conv2D**. Il modello viene addestrato usando la **mean squared error (MSE)** per predire il frame successivo.

Suddivisione del dataset utilizzata dal progetto:

- 8.000 sequenze per il training
- 1.000 sequenze per la validation
- 1.000 sequenze per il test
