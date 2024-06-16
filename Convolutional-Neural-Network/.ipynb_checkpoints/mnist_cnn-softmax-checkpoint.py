### CNN for categorial classification (using 0,1 from MNIST)
#############################################################

import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import random
import math
import matplotlib.pyplot as plt

from dense import Dense
from convolutional import Convolutional
from pooling import MaxPooling
from reshape import Reshape
from activations import Sigmoid
from softmax import Softmax
from losses import binary_cross_entropy_softmax, binary_cross_entropy_derivative_softmax
from network import train, predict

#Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 20)

### Network with Softmax Layer
# network = [
#     Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5, mode='valid'),
#     Sigmoid(log=False),
#     Reshape(input_shape=(5, 26, 26), output_shape=(3380, 1)),
#     Dense(3380, 100),
#     Sigmoid(log=False),
#     Dense(100, 2),
#     Softmax(log=False)
# ]

### Network with Softmax Layer and MaxPooling
network = [
    Convolutional(input_shape=(1, 28, 28), kernel_size=5, depth=5, mode='valid'),
    MaxPooling(pool_size=(2,2), stride=2),
    Sigmoid(log=False),
    Reshape(input_shape=(5, 12, 12), output_shape=(720, 1)),
    Dense(720, 100),
    Sigmoid(log=False),
    Dense(100, 2),
    Softmax(log=False)
]

# Training the network
train(
    network,
    binary_cross_entropy_softmax,
    binary_cross_entropy_derivative_softmax,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.01
)

for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
