# CT Medical Image Classification System (T1-T4 Staging) 
Deep learning-based CT image classification tool supporting ResNet/DenseNet/VGG architectures for T1-T4 stage prediction

[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Architecture Diagram](docs/architecture.png) <!-- Recommended to add visual abstract -->

## Table of Contents
- [Key Features](#key-features)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Configuration Options](#-configuration-options)
- [Performance Evaluation](#-performance-evaluation)
- [Contributing](#-contributing)

## Key Features
- Supports three mainstream CNN architectures: ResNet-152, DenseNet-169, VGG-19
-  Automatic generation of classification metrics (Precision/Recall/F1-score)
-  Real-time prediction logging during training
-  GPU acceleration support (automatic device detection)
- Comprehensive data augmentation pipeline

##  Getting Started

### System Requirements
- Python 3.8+
- CUDA 11.3+ (Recommended for GPU acceleration)
- Minimum 8GB RAM

### Installation
1. Clone repository:
```bash
git clone https://github.com/yourname/ct-classification.git
cd ct-classification

1.Install dependencies:
pip install -r requirements.txt

2.Prepare data:
data/
├── images/      # Store all CT images (JPEG/PNG format)
└── labels.csv   # CSV file with columns: image_name, label

Basic Usage
Train ResNet-152 model:
python src/train.py \
  --data_dir ./data/images \
  --csv_file ./data/labels.csv \
  --model resnet152 \
  --batch_size 64 \
  --lr 0.001 \
  --epochs 300

Project Structure
ct-classification/
├── data/                # Raw imaging data (Git ignored)
├── models/              # Trained model weights
├── outputs/             # Training outputs
│   ├── predictions/     # Prediction results in CSV format
├── src/                 # Source code
│   ├── dataset.py       # Data loading & preprocessing
│   ├── model.py         # Model initialization
│   └── train.py         # Main training pipeline
├── requirements.txt     # Dependency list
└── README.md            # Documentation
Configuration Options
Customize training through command-line arguments:
python src/train.py --help  # Display all available parameters

Essential parameters:
  --model        Architecture selection (resnet152/densenet169/vgg19) [default: resnet152]
  --batch_size   Batch size [default: 32]
  --lr           Learning rate [default: 0.001]
  --epochs       Training epochs [default: 10]
  --augment      Enable data augmentation [default: True]
  --seed         Random seed [default: 42]

Performance Evaluation
Training outputs include:

.pth model files in models/
Prediction CSV files in outputs/predictions/
Detailed classification report:
              precision    recall  f1-score   support

         T1       0.92      0.91      0.92       203
         T2       0.89      0.93      0.91       198
         T3       0.95      0.88      0.92       210
         T4       0.90      0.94      0.92       189

    accuracy                           0.91       800
   macro avg       0.91      0.91      0.91       800
weighted avg       0.91      0.91      0.91       800


