### CNN for categorial classification (using 0,1,2 from MNIST)
#############################################################

import numpy as np
from keras.datasets import mnist
import random
import matplotlib.pyplot as plt

from dense import Dense
from convolutional import Convolutional
from pooling import MaxPooling
from reshape import Reshape
from activations import Sigmoid
from softmax import Softmax
from losses import categorical_cross_entropy, categorical_cross_entropy_derivative
from network import train, predict

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preview data
def display_image_and_matrix(index, data, label):
    image = data[index]
    label = label[index]

    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

    height, width = image.shape
    depth = 1  # By default MNIST
    print(f"Height: {height}, Width: {width}, Depth: {depth}")

# Function to preprocess data
def preprocess_data(x, y, limit, num_classes):
    indices = []
    for c in range(num_classes):
        indices.extend([i for i, label in enumerate(y) if label == c][:limit])
    random.shuffle(indices)

    x = [x[i] for i in indices]
    y = [y[i] for i in indices]

    x_reshaped = []
    for item in x:
        reshaped = item.reshape(-1, 1) / 255.0
        x_reshaped.append(reshaped)
    x = x_reshaped

    y_encoded = np.zeros((len(y), num_classes))
    for i, item in enumerate(y):
        y_encoded[i, item] = 1
    y = y_encoded

    return x, y

# Preprocess data for 0, 1, and 2 labels
x_train, y_train = preprocess_data(x_train, y_train, 100, 3)
x_test, y_test = preprocess_data(x_test, y_test, 100, 3)

### Network with Softmax Layer
network = [
    Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5, mode='valid'),
    Sigmoid(log=False),
    Reshape(input_shape=(5, 26, 26), output_shape=(3380, 1)),
    Dense(3380, 100),
    Sigmoid(log=False),
    Dense(100, 3),  # 3 classes: 0, 1, 2
    Softmax(log=False)
]

# Training the network
train(
    network,
    categorical_cross_entropy,
    categorical_cross_entropy_derivative,
    x_train,
    y_train,
    epochs=1,
    learning_rate=0.1
)
