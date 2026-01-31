# Quantized Mushroom Classification 🍄

When foraging for mushrooms, many people struggle to identify the type of mushroom they have found. There is a wide variety of mushrooms one might stumble across in the woods, and many mushrooms have similar appearances making them difficult to distinguish. It is important to know whether or not the mushroom is edible. If it is, one would also like to know if it is tasty. To accomplish this, we plan to build a model capable of identifying ~50 species of mushrooms. The output will be top five species of mushroom, whether it’s edible and the certainty of the prediction. Furthermore, cell service is often unavailable in the forest, so it is beneficial to have a system which works offline. For this purpose, we will also attempt to compress through weight quantization.

---

## Table of Contents

1. [Implementation Plan](#implementation-plan)
2. [Project Setup](#project-setup)  

---

## Implementation Plan

The primary goal of this project is to investigate how model compression, specifically weight quantization, influences classification performance in a fine-grained, safety-relevant image recognition task. The focus of the evaluation lies in understanding the trade-off between model accuracy, reliability, and deployment efficiency, particularly in offline scenarios where computational resources are limited.

The implementation uses python 3.11 and tensorflow for better compatibility with the chosen models (EfficientNet-V2).

### Data Acquisition

The dataset is loaded via API from [iNaturalist](https://www.inaturalist.org/) and consists of labeled mushroom images covering approximately 50 species. To reflect realistic deployment constraints and reduce computational cost, lower-resolution images are used throughout the experiments. This choice also allows us to assess how well modern convolutional architectures generalize under constrained input quality.

### Data Loading and Preprocessing

The data loader performs the following preprocessing steps:

- Rescaling input images to a fixed resolution
- Padding to preserve aspect ratio where necessary
- Data augmentation (random flips, rotations, zoom, and contrast adjustments) to improve generalization and robustness

These steps are applied consistently across all models to ensure fair comparison.

### Models for Comparison

To evaluate both accuracy and efficiency, several architectures and training strategies are compared:

- EfficientNet-V2-M (fine-tuned): 
EfficientNet-V2-M is selected as the primary model due to its strong performance on lower-resolution images and its suitability for transfer learning. Larger variants (V2-L / V2-XL) are excluded as they typically require high-resolution inputs to fully leverage their capacity.

- EfficientNet-V2-M (fine-tuned + quantized):
The same fine-tuned model is further compressed using quantization-aware training (QAT) to evaluate the impact of weight quantization on accuracy, calibration, and inference speed.

- Baseline model (ResNet):
A standard ResNet architecture is used as a baseline. Only the final classification layer is replaced and trained, without full fine-tuning, to provide a lower-bound reference for performance.

- Optional: EfficientNet-V2-S (fine-tuned):
If time permits, a smaller EfficientNet-V2-S model is fine-tuned to analyze performance–efficiency trade-offs for even more constrained deployment scenarios and to compare the quantized model with the smaller version of the same model.

### Training Procedure

Training is performed in two stages:

- Fine-tuning phase
The classifier head is trained first, followed by partial unfreezing of the backbone to adapt high-level features to the mushroom classification task.

- Quantization-aware training (QAT)
After fine-tuning, QAT is applied to the EfficientNet-V2-M model to ensure a fair comparison between the full-precision and quantized versions.

All experiments are tracked using Weights & Biases (W&B) to log hyperparameters, training curves, evaluation metrics, and model artifacts.

### Evaluation Metrics

Model performance is evaluated using the following metrics:

- Top-1 accuracy
- Top-5 accuracy (crucial given the large number of visually similar species)
- Precision, Recall, and F1-score, with particular emphasis on the edible vs. poisonous distinction
- Model size (MB) to quantify compression benefits
- Inference time on CPU, reflecting realistic offline deployment conditions
- Accuracy degradation after quantization, measuring the cost of compression

### Additional Analysis

Given the safety-critical nature of mushroom identification, confidence reliability is also evaluated:

- Calibration metrics such as Expected Calibration Error (ECE)
- Analysis of prediction confidence to assess whether the model’s certainty aligns with its accuracy

Reliable confidence estimates are especially important when advising users about edibility, where incorrect but overconfident predictions could have severe consequences.

---
## Project Setup

Clone the repository:

```bash
git clone https://github.com/ninaimmenroth/quantized-mushroom-classification
cd quantized-mushroom-classification
```

The `requirements.txt` file is already included in the repo.


### Virtual Environment Setup

#### Mac / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### Windows (CMD)

```bash
python3.11 -m venv .venv
.venv\Scripts\activate.bat
```

#### Windows (PowerShell)

```bash
python3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

After activation, your prompt should show:

```
(.venv) $
```


### Install Dependencies

With the virtual environment active:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Optional GPU acceleration on Mac (M1/M2/M3)**:

```bash
pip install tensorflow-metal
```


### Verify Installation

Check TensorFlow version and available devices:

```bash
python - <<EOF
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())
EOF
```

You should see your CPU and optionally GPU devices listed.


### Running the Training Script

```bash
python train_efficientnet_v2.py
```

- Training automatically loads images from:
  - `data/train/` → training images
  - `data/val/` → validation images
- The script handles:
  - Resize with padding (aspect ratio preserved)
  - Data augmentation (rotation, zoom, contrast, flip)
  - Two-stage training: head training + fine-tuning


### Project Directory Structure

```
mushroom-classification/
├── train_efficientnet_v2.py
├── requirements.txt
├── data/
│   ├── train/
│   │   ├── species1/
│   │   ├── species2/
│   │   └── ...
│   └── val/
│       ├── species1/
│       ├── species2/
│       └── ...
└── .venv/   # virtual environment folder
```


### Optional GPU Setup

#### Mac (Apple Silicon)
```bash
pip install tensorflow-metal
```

#### Windows / Linux (NVIDIA)
- Install CUDA + cuDNN as per TensorFlow GPU guide
- TensorFlow will automatically detect GPU at runtime


### Deactivating the Environment

When finished:

```bash
deactivate
```

Your shell prompt will return to normal.


---

