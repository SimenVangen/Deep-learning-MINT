# Deep Learning Model Comparison

This repository contains code for comparing the performance of different deep learning models on the MNIST dataset. The project explores three models: Multi-Layer Perceptron (MLP), 1-D Convolutional, and 2-D Convolutional models. The performance of each model is tracked using Weight & Biases (wandb) for loss and accuracy.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Hyperparameters](#hyperparameters)
- [Data Augmentation](#data-augmentation)
- [Models](#models)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Project Description

In this project, we evaluate the performance of different deep learning models on the MNIST dataset. Each model is implemented and trained using PyTorch, and their performance is tracked with Weight & Biases (wandb).

## Installation

To run this project, you need to install the required libraries and dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

# Hyperparameters
The project uses various hyperparameters to configure the training process:

Input Size: 784
Hidden Size: 32
Number of Classes: 10
Number of Epochs: 6
Batch Size: 100
Learning Rate: 0.01

# Data Augmentation
To improve model resilience, the project applies three data augmentation techniques:

Horizontal Flip at Random
Random Rotation
Color Jittering
Models
Three different models are implemented in this project:

# MLP Model:

Input Layer: 784 neurons
Two Hidden Layers, each with 32 neurons
Activation Functions: Leaky ReLU, ReLU, Tanh
Output Layer: 10 neurons for class prediction
1-D Convolutional Model:

1-D Convolutional Layers
ReLU Activation
Max-Pooling Layers
Fully Connected Layers
2-D Convolutional Model:

2-D Convolutional Layers
ReLU Activation
Max-Pooling Layers
Fully Connected Layers
Training
The models are trained using different optimizers, such as Adam, RMSprop, and SGD. The project logs training loss and accuracy during each epoch using wandb.

# Results
The project records the loss and accuracy of each model, comparing their performance across epochs. Results can be visualized in Weight & Biases (wandb).
