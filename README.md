# ML Monitoring — FashionMNIST CNN
### FSDL 2022 — Lab 08 (Modernized)

A hands-on walkthrough of monitoring a deployed machine learning model in production, built as a modernized version of FSDL 2022's Lab 08. This project simulates the core building blocks of ML monitoring using a small CNN trained on FashionMNIST, without relying on any external monitoring platforms.

## Overview

Once a model is deployed, training accuracy alone doesn't tell you how it's actually performing in the real world. This notebook demonstrates how to track that in practice by logging predictions, measuring latency, flagging low-confidence outputs, detecting data drift, and simulating user feedback to catch model errors before they compound.

## What's Inside

- **Model**: A simple CNN (`SimpleCNN`) — two convolutional blocks followed by two fully connected layers — trained on FashionMNIST in a companion notebook (Lab 07) and loaded here via saved weights.
- **Prediction Logging**: Every inference is timed and logged with its predicted class, confidence score, and latency, then aggregated into a pandas DataFrame for analysis.
- **Performance Dashboard**: Summary statistics (accuracy, average/median/max/min latency, average confidence) plus visualizations of latency trends and confidence distribution.
- **Confidence Monitoring**: Flags predictions below a configurable confidence threshold and raises an alert if too many predictions fall into that low-confidence bucket.
- **Data Drift Simulation**: Artificially shifts the input distribution (darkened images + noise) to mimic real-world data drift, then detects it by comparing pixel-intensity distributions between training and "production" data.
- **Feedback & Error Analysis**: Simulates user feedback on predictions, logs incorrect ones, and surfaces which classes are most frequently misclassified — a basic human-in-the-loop retraining signal.
- **Unified Monitoring Dashboard**: Consolidates all metrics (accuracy, confidence, latency, drift, errors) into a single summary view with diagnostic plots.

## Requirements

```
torch
torchvision
pandas
matplotlib
tqdm
```

## Usage

1. Train the model in the companion Lab 07 notebook and save its weights:
   ```python
   torch.save(model.state_dict(), "simple_cnn_weights.pth")
   ```
2. Place `simple_cnn_weights.pth` in the same directory as this notebook (or update `WEIGHTS_PATH`).
3. Run the notebook top to bottom. If no weights file is found, the notebook falls back to an untrained model purely to demonstrate the monitoring pipeline mechanics.

## Background

Originally based on FSDL (Full Stack Deep Learning) 2022's Lab 08, this version was rebuilt to run on current, actively maintained package versions and simplified into a self-contained notebook suitable for demonstrating monitoring concepts independent of the original course infrastructure.

