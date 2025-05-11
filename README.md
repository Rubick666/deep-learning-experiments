# Deep Learning Experiments

This repository contains several deep learning projects implemented using TensorFlow and Keras. The projects cover a range of tasks including image classification, text classification, sequence modeling, and autoencoders.

## Projects Included

### 1. Simple AutoEncoder
- Implements an autoencoder for image reconstruction.
- Demonstrates encoding and decoding of input images.
- Explores dimensionality reduction and feature learning.

### 2. MNIST Classification
- `MNIST_with_Conv(functionalAPI).ipynb`: CNN model built using Keras Functional API.
- `MNIST_With_MLP(Sequential).ipynb`: Multi-layer Perceptron using Keras Sequential API.

### 3. Text Classification on IMDB Dataset
- `keras_imdb_withLSTM.ipynb`: Sentiment analysis using LSTM layers.
- `keras_imdb_withSimpleEmbedding.ipynb`: Simple embedding layer for text classification.
- `keras_imdb_withSimpleRNN.ipynb`: Sentiment analysis using simple RNN layers.

### 4. Regression on Housing Prices
- Predicts housing prices using deep learning regression models.

## Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib

## How to Run
1. Open the desired notebook in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
2. Run all cells sequentially.
3. (Optional) To run locally, install dependencies with:

    ```
    pip install -r requirements.txt
    ```

## Results
- Achieved high accuracy on MNIST classification tasks.
- Demonstrated effective text classification on IMDB dataset.
- Autoencoder successfully reconstructs input images.
- Regression model predicts housing prices with reasonable accuracy.

## Folder Structure
deep-learning-experiments/
├── Simple_AutoEncoder.ipynb
├── MNIST_with_Conv(functionalAPI).ipynb
├── MNIST_With_MLP(Sequential).ipynb
├── keras_imdb_withLSTM.ipynb
├── keras_imdb_withSimpleEmbedding.ipynb
├── keras_imdb_withSimpleRNN.ipynb
├── RegressionOnHousingPrices.ipynb
├── requirements.txt
└── README.md

## Author
- [Amirfarhad](https://github.com/Rubick666)
