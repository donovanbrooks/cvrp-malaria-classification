# Malaria Cell Image Classification Using CNN

## Overview
This project uses a Convolutional Neural Network (CNN) to classify cell images as either Parasitized or Uninfected.

The project was completed as part of the Computer Vision Research Program (CVRP 2026).

## Technologies
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-Learn

## Dataset
The dataset contains microscopic cell images from two classes:
- Parasitized
- Uninfected

Images were resized to 64x64 grayscale before training.

## CNN Architecture
- Conv2D (32 filters, 3x3)
- MaxPooling2D
- Conv2D (64 filters, 3x3)
- MaxPooling2D
- Flatten
- Dense (128)
- Dense (1, Sigmoid)

## Training Parameters
- Batch Size: 20
- Epochs: 35
- Optimizer: Adam
- Loss Function: Binary Crossentropy

## Results
Final Test Accuracy: 86.37%

## Acknowledgments
This project was completed as part of the Computer Vision Research Program (CVRP 2026) using project requirements and instructional guidance provided by Professor Sahin Ismet.
