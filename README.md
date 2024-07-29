**# Deep-Learning-Model-Training-and-Evaluation**
# Deep Learning Model Training and Evaluation

This project demonstrates the training and evaluation of various deep learning models using Python and PyTorch. The script can handle different neural network architectures and includes options for data preprocessing and dimensionality reduction using PCA.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Methods](#methods)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Arguments](#arguments)

## Requirements

- Python 3.x
- NumPy
- PyTorch
- torchinfo

## Installation

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/deep-learning-model-training.git
cd deep-learning-model-training
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with the following command:

```bash
python main.py --data <data_path> --nn_type <nn_type> [additional arguments]
```

Example:

```bash
python main.py --data dataset --nn_type cnn --use_pca --pca_d 50
```

## Methods

The following neural network architectures are available:

- `mlp`: Multi-Layer Perceptron
- `cnn`: Convolutional Neural Network
- `transformer`: Vision Transformer

## Data Preparation

Data can be prepared and preprocessed using the following steps:

1. Loading and flattening images into vectors
2. Creating a validation set from the training data
3. Normalizing the data
4. Optional dimensionality reduction using PCA

## Training and Evaluation

The script trains the specified model on the training data and evaluates it on the validation or test data. The evaluation metrics are:

- Accuracy
- Macro F1-score

## Arguments

- `--data`: Path to your dataset.
- `--nn_type`: The neural network architecture to use (`mlp` | `transformer` | `cnn`).
- `--nn_batch_size`: Batch size for neural network training.
- `--device`: Device to use for training (`cpu` | `cuda` | `mps`).
- `--use_pca`: Use PCA for feature reduction.
- `--pca_d`: Number of principal components for PCA.
- `--lr`: Learning rate for the optimizer.
- `--max_iters`: Maximum number of iterations (epochs) for training.
- `--test`: Use the test set for evaluation instead of creating a validation set.

Feel free to add more arguments if needed!

## Contributing

If you want to contribute to this project, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
