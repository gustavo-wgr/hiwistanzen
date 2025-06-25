
<h1 align="center">Punching Data Augmentation Toolkit</h1>

## Dataset Overview

In this repository, datasets are named following the structure:

```
{signal_type}{material_type}{thickness}
```

For example: `RPDC197`

### Explanation

- **signal_type**:  
  This repository focuses exclusively on `RP` signals

- **material_type**:  
  Two types are used based on the training phase:
  - `HC`: used for **pretraining** models.
  - `DC`: used for **fine-tuning** and **evaluation**.

- **thickness**:  
  The possible values used in this project are:  
  `{185, 188, 191, 194, 197}`

### Structure

Each dataset (e.g., `RPDC197`) contains approximately **5000 samples**.  
Each sample is:
- A time series with **2800 datapoints**.
- Stored as a `.pkl` file with a single label.
- Named as `RXX_YYY.pkl`, where:
  - `RXX` (e.g., `R05`) is the label.
  - `YYY` (e.g., `I300`) is an identifier (not used in labeling).

The set of labels used is:
```
{R05, R10, R15, R20, R25, R30, R35, R40, R45, R50}
```

These datasets are used in various combinations across pretraining, augmentation, and evaluation pipelines.

## Experimental Setups

To explore how the amount of available labeled data impacts classifier performance, we subdivide datasets into eight training splits of increasing size:

- `train_20`, `train_50`, `train_100`, `train_200`, `train_300`,  
  `train_400`, `train_500`, `train_600`

Each `train_k` set is **cumulative**, meaning it includes all samples from the previous split plus additional samples. This allows us to examine performance scaling as more data becomes available.

A dedicated validation set `val_1000` is used for performance evaluation. Validation is conducted across five dataset variants (RPDC185–RPDC197), simulating real-world generalization challenges across material thickness variations.

### Classifier Architecture

The model used in all experiments is a 1D convolutional neural network (CNN) with the following structure:

- **Input**: A single-channel 1D signal of length 2800.
- **Three convolutional blocks**:
  - Each with `Conv1D → BatchNorm → ReLU → MaxPool`.
  - Channel progression: 1 → 16 → 32 → 64.
- **Fully connected head**:
  - Flatten → Dense(128) → ReLU → Dropout(0.5) → Output layer (10 classes).

This architecture is implemented using PyTorch and trained with the following configuration:
- **Optimizer**: Adam (`lr=1e-3`, `weight_decay=1e-5`)
- **Scheduler**: StepLR (decay every 15 epochs)
- **Loss function**: CrossEntropyLoss
- **Epochs**: 50 (baseline); adjusted per method
- **Gradient clipping**: max norm = 1.0
- **Evaluation**: Accuracy averaged over 10 different random seeds

## Results
<details style="margin: 1.5em 0; padding: 1em; border: 1px solid #ddd; border-radius: 6px; background-color: #f9f9f9;">
  <summary style="font-size: 1.1em; font-weight: 600; cursor: pointer;">
    Baseline
  </summary>
  <div align="center" style="margin-top: 1em;">
    <img src="https://i.imgur.com/gZGy95c.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
  </div>
</details>
<details style="margin: 1.5em 0; padding: 1em; border: 1px solid #ddd; border-radius: 6px; background-color: #f9f9f9;">
  <summary style="font-size: 1.1em; font-weight: 600; cursor: pointer;">
    Jitter Augmentation
  </summary>
  <div align="center" style="margin-top: 1em;">
    <img src="https://i.imgur.com/gqMoy77.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
  </div>
</details>
<details style="margin: 1.5em 0; padding: 1em; border: 1px solid #ddd; border-radius: 6px; background-color: #f9f9f9;">
  <summary style="font-size: 1.1em; font-weight: 600; cursor: pointer;">
    SHOT vs Coral Pretraining
  </summary>
  <div align="center" style="margin-top: 1em;">
    <img src="https://i.imgur.com/6oVkmdQ.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <img src="https://i.imgur.com/ioQ8WmT.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <img src="https://i.imgur.com/aoqI5hQ.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <img src="https://i.imgur.com/4Hattro.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <img src="https://i.imgur.com/PpS4bPm.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
  </div>
</details>
<details style="margin: 1.5em 0; padding: 1em; border: 1px solid #ddd; border-radius: 6px; background-color: #f9f9f9;">
  <summary style="font-size: 1.1em; font-weight: 600; cursor: pointer;">
    GAN Augmentation
  </summary>
  <div align="center" style="margin-top: 1em;">
    <img src="https://i.imgur.com/XZPKUgc.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">

  </div>
</details>
<details style="margin: 1.5em 0; padding: 1em; border: 1px solid #ddd; border-radius: 6px; background-color: #f9f9f9;">
  <summary style="font-size: 1.1em; font-weight: 600; cursor: pointer;">
    Simple Pretraining
  </summary>
  <div align="center" style="margin-top: 1em;">
    <img src="https://i.imgur.com/CvEMc5p.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">

  </div>
</details>
<details style="margin: 1.5em 0; padding: 1em; border: 1px solid #ddd; border-radius: 6px; background-color: #f9f9f9;">
  <summary style="font-size: 1.1em; font-weight: 600; cursor: pointer;">
    Combined Augmentation
  </summary>
  <div align="center" style="margin-top: 1em;">
    <img src="https://i.imgur.com/JsZxvxw.jpeg" alt="Fully Supervised Diagram" style="max-width: 100%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">

  </div>
</details>
