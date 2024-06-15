from activations.activation import Activation
from math import tanh as math_tanh
from layers.layer import Layer
import math
from helpers.algebra_helper import Algebra

# Must work with 1D vector (for matrix dense layer)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return math.tanh(x)

        def tanh_derivative(x):
            return 1 - math.tanh(x) ** 2

        super().__init__(tanh, tanh_derivative)

class Sigmoid(Activation):
    def __init__(self, log=False):
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))  
        
        def sigmoid_derivative(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_derivative, log=log)