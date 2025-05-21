# CT Medical Image Classification System (T1-T4 Staging)  

![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

Deep learning-based CT image classification tool supporting ResNet/DenseNet/VGG architectures for T1-T4 stage prediction.

![Architecture Diagram](docs/architecture.png)

## Table of Contents
- [✨ Key Features](#-key-features)
- [🚀 Getting Started](#-getting-started)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration Options](#️-configuration-options)
- [📊 Performance Evaluation](#-performance-evaluation)
- [🤝 Contributing](#-contributing)

## ✨ Key Features
- **Multi-Architecture Support**: Choose between ResNet-152, DenseNet-169, or VGG-19
- **Comprehensive Metrics**: Automatic generation of precision/recall/F1-score reports
- **Training Visualization**: Real-time prediction logging and progress tracking
- **GPU Acceleration**: Automatic detection and utilization of available GPUs
- **Advanced Augmentation**: Built-in pipeline for robust model training
- **Model Management**: Save/load functionality with training checkpointing

## 🚀 Getting Started

### System Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.3+ (Recommended for GPU acceleration)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: SSD with minimum 10GB free space

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourname/ct-classification.git
cd ct-classification

1.Install dependencies:
pip install -r requirements.txt

2.Prepare your data structure:
data/
├── images/       # Store all CT images (JPEG/PNG format)
└── labels.csv    # CSV file with columns: [image_name, label]

Basic Usage
Train a model with default parameters (ResNet-152):
python src/train.py \
  --data_dir ./data/images \
  --csv_file ./data/labels.csv

For custom training (example with DenseNet-169):
python src/train.py \
  --model densenet169 \
  --batch_size 64 \
  --lr 0.001 \
  --epochs 50 \
  --save_dir ./custom_models

📁 Project Structure
ct-classification/
├── data/                # Raw imaging data (.gitignore)
├── models/              # Trained model checkpoints
├── outputs/             # Training outputs
│   ├── predictions/     # Prediction results (CSV)
│   └── visualizations/  # Confusion matrices
├── src/                 # Source code
│   ├── dataset.py       # Data loading & preprocessing
│   ├── model.py         # Model initialization
│   ├── train.py         # Main training pipeline  
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

⚙️ Configuration Options
Run python src/train.py --help to see all available parameters:

Parameter	Description	Default	Options
--model	Model architecture	resnet152	resnet152/densenet169/vgg19
--batch_size	Training batch size	32	Integer > 0
--lr	Initial learning rate	0.001	Float > 0
--epochs	Number of training epochs	10	Integer > 0
--augment	Enable data augmentation	True	True/False
--seed	Random seed for reproducibility	42	Integer
--save_dir	Directory to save models	./models	Valid path
📊 Performance Evaluation
Example classification report:
              precision    recall  f1-score   support

         T1       0.92      0.91      0.92       240
         T2       0.89      0.93      0.91       235
         T3       0.95      0.88      0.92       238
         T4       0.90      0.94      0.92       240

    accuracy                           0.91       953
   macro avg       0.91      0.91      0.91       953
weighted avg       0.91      0.91      0.91       953

Outputs include:

1.Trained models (.pth) in specified save directory
2.Prediction results in CSV format
3.Training logs and metrics
4.Visualizations (confusion matrices)

🤝 Contributing
We welcome contributions! Please follow these steps:

1.Fork the repository
2.Create your feature branch (git checkout -b feature/your-feature)
3.Commit your changes (git commit -am 'Add some feature')
4.Push to the branch (git push origin feature/your-feature)
5.Open a Pull Request
