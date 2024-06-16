# CNN and Neural Network for MNIST

This project implements a Convolutional Neural Network (CNN) and a Neural Network (NN) from scratch. The project is organized into multiple modules for clarity and maintainability.

### How to Run 
cd /path/to/Convolutional-Neural-Network

python mnist_cnn.py

python mnist_nn.py


### File Descriptions

- `models/`: Contains model-related functions.
  - `network.py`: Define the Neural Network process, including layers and foward-backpropagate frow from layers to layers.

- `layers/`: Contains various layer implementations.
  - `layer.py`: Defines the base `Layer` class which is the interface for all layers.
  - `dense.py`: Implements the `Dense` layer.
  - `convolutional.py`: Implements the `Convolutional` layer.
  - `pooling.py`: Implements `MaxPooling` layer.
  - `reshape.py`: Implements reshaping layers.

- `activations/`: Contains activation functions.
  - `activation.py`: Base class for activations.
  - `activations.py`: Specific activation functions like `Sigmoid`.

- `helpers/`: Contains helper functions for algebraic and convolutional operations.
  - `algebra_helper.py`: Helper functions for algebraic operations.
  - `convolutional_helper.py`: Helper functions for convolutional operations.

- `utils/`: Contains utility functions.
  - `losses.py`: Defines loss functions and their derivatives (including Binary-Cross-Entroy and MSE).

- `mnist/`: Contains scripts to run the MNIST CNN and NN models.
  - `mnist_cnn.py`: Script to run the CNN on the MNIST dataset.
  - `mnist_nn.py`: Script to run a simple NN on the MNIST dataset.

- `main.py`: Entry point for the application.

### Prerequisites

Ensure you have the following packages installed (only used to quickly access MNIST dataset):
- numpy
- keras
- matplotlib

You can install these packages using pip:

pip install numpy keras matplotlib
