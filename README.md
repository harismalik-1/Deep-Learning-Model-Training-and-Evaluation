# Vision Transformer & Neural Networks on MNIST

This repository demonstrates how to:

1. **Load and preprocess image data** (e.g., MNIST or a custom dataset).  
2. **Optionally perform PCA** for dimensionality reduction.  
3. **Train different deep learning models** (MLP, CNN, and a custom Vision Transformer) on the data.  
4. **Evaluate model performance** with metrics like accuracy and macro-F1.

---

## Table of Contents

- [Overview](#overview)
- [Key Files and Structure](#key-files-and-structure)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Example Runs](#example-runs)
- [Vision Transformer Details](#vision-transformer-details)
- [Trainer Class](#trainer-class)
- [Dependencies and Installation](#dependencies-and-installation)
- [All Code in One Place](#all-code-in-one-place)
  - [main.py](#mainpy)
  - [example_vit_mnist.py](#example_vit_mnistpy)
  - [src/data.py](#srcdatapy)
  - [src/methods/pca.py](#srcmethodspcapy)
  - [src/methods/deep_network.py](#srcmethodsdeep_networkpy)
  - [src/utils.py](#srcutilspy)
- [License](#license)
- [Notes](#notes)

---

## Overview

This project offers a **flexible framework** for exploring various neural network architectures on image classification tasks:

- **MLP (Multi-Layer Perceptron)** and **CNN (Convolutional Neural Network)** for standard approaches.
- **MyViT** – a Vision Transformer implementation that splits images into patches and processes them via multi-headed self-attention.
- A **Trainer** class that simplifies data loading, training, validation, and metrics computation.

Additionally, the code supports **PCA-based** dimensionality reduction, letting you experiment with feature compression.

---

## Key Files and Structure

Although below we provide a single README with all code, a typical file structure could look like:


- **main.py**  
  Orchestrates data loading, validation-split creation, PCA usage, and training of the specified model (`mlp`, `cnn`, or `transformer`).

- **example_vit_mnist.py**  
  A demo training script specifically using the Vision Transformer (`MyViT`) on MNIST via `torchvision`.

- **src/data.py**  
  Handles dataset loading and initial preprocessing.

- **src/methods/pca.py**  
  Implements a simple PCA class for feature reduction.

- **src/methods/deep_network.py**  
  Contains the neural network architectures (`MLP`, `CNN`, and `MyViT`) and the `Trainer` class for model training.

- **src/utils.py**  
  Utility functions for normalization, accuracy, macro-F1, and more.

---

## Usage

### Command-Line Arguments

Use `main.py` with arguments to control data paths, model type, training parameters, and more. Key flags:

- `--data`: Path to your dataset (default: `"dataset"`).  
- `--nn_type`: Model architecture to use (`mlp`, `cnn`, or `transformer`).  
- `--nn_batch_size`: Batch size (default: `64`).  
- `--device`: Either `"cpu"`, `"cuda"`, or `"mps"` (default: CPU).  
- `--use_pca`: If set, applies PCA to the data.  
- `--pca_d`: Number of principal components (default: `100`).  
- `--lr`: Learning rate (default: `0.1`).  
- `--max_iters`: Number of training epochs (default: `15`).  
- `--test`: If set, trains on full training data (no validation set) and generates predictions for the test set.

```bash
python main.py --help
