import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import random

from dense import Dense
from convolutional import Convolutional
from pooling import MaxPooling
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_derivative
from network import train, predict
import matplotlib.pyplot as plt

#Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Preview data
def display_image_and_matrix(index, data, label):
    image = data[index]
    label = label[index]

    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

    height, width = image.shape
    depth = 1 #By default MNIST
    print(f"Height: {height}, Width: {width}, Depth: {depth}")

# display_image_and_matrix(0, x_train, y_train)

def preprocess_data(x, y, limit):
    # Collect label 0 and 1 only
    zero_index = [i for i in range(len(y)) if y[i] == 0][:limit]
    one_index = [i for i in range(len(y)) if y[i] == 1][:limit]
    all_indices = zero_index + one_index
    random.shuffle(all_indices)
    
    # Init data with label set
    x = [x[i] for i in all_indices]
    y = [y[i] for i in all_indices]
    
    # Reshape x
    x_reshaped = []
    for item in x:
        reshaped = []
        for row in item:
            reshaped.extend(row)
        x_reshaped.append([[float(pixel) / 255] for pixel in reshaped])
    x = x_reshaped
    
    # Encode y
    # Initializes an empty list y_encoded.
    # For each label in y, appends a one-hot encoded representation:
    # If the label is 0, append [[1], [0]] (indicating the first class).
    # If the label is 1, append [[0], [1]] (indicating the second class).
    y_encoded = []
    for item in y:
        if item == 0:
            y_encoded.append([[1], [0]])
        else:
            y_encoded.append([[0], [1]])
    y = y_encoded
    
    return x, y

x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# network = [
#     Convolutional(input_shape=(1, 28, 28), kernel_size=3, depth=5, mode='valid'),
#     Sigmoid(log=False),
#     Reshape(input_shape=(5, 26, 26), output_shape=(3380, 1)), #Output is (5*26*26=3380, 1)
#     Dense(5 * 26 * 26, 100),
#     Sigmoid(log=False),
#     Dense(100, 2),
#     Sigmoid(log=False)
# ]

network = [
    Convolutional(input_shape=(1, 28, 28), kernel_size=5, depth=5, mode='valid'), #Down to (5,24,24)
    MaxPooling(pool_size=(2,2), stride=(2,2)), #Down to (5,12, 12)
    Sigmoid(log=False),
    Reshape(input_shape=(5, 12, 12), output_shape=(720, 1)), 
    Dense(720, 100),
    Sigmoid(log=False),
    Dense(100, 2),
    Sigmoid(log=False)
]


train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_derivative,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")