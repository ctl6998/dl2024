import random
from layers.layer import Layer
from helpers.algebra_helper import Algebra

# Fully-connected layer with input (ix1) and output (jx1)
class Dense(Layer):
    ################################## INITIALIZE WEIGHTS & BIASES ################################## 
    ### Initialize weights: matrix size (jxi)
    ### Initialize bias: matrix size (jx1)
    ### j row as output size (jx1)
    ### i columns as input size (ix1)
    def __init__(self, input_size, output_size):
        self.weights = [[random.gauss(0, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.bias = [[random.gauss(0, 1)] for _ in range(output_size)]
    
    ################################## FOWARDING ##################################
    ### Forward compute and return matrix Y = W.X + B
    ### Y-(jx1) = W-(jxi) (dot) X-(ix1) + B-(jx1)
    def forward(self, input):
        super().forward(input)
        # print("::::::::::::::::DENSE/FULLY-CONNECTED LAYER FOWARDING::::::::::::::::")
        self.input = input
        self.output = Algebra.matrix_add(Algebra.matrix_mult(self.weights, self.input), self.bias)
        return self.output
    
    ################################## BACKWARDING ##################################
    ### output_gradient is derivative dE/dY, we proved dE/dY = dE/dB (Bias gradient) (matrix: jx1)
    ### weights_gradient (matrix: jxi) dE/dW = dE/dY . X-(transposed) 
    ### input_gradient (matrix: ix1) = W-(transposed) . dE/dY 
    def backward(self, output_gradient, learning_rate):
        super().backward(output_gradient, learning_rate)
        # print("::::::::::::::::DENSE LAYER BACKPROPAGATION::::::::::::::::")
        weights_gradient = Algebra.matrix_mult(output_gradient, Algebra.transpose(self.input))
        input_gradient = Algebra.matrix_mult(Algebra.transpose(self.weights), output_gradient)
        self.weights = Algebra.matrix_sub(self.weights, Algebra.scalar_mult(weights_gradient, learning_rate))
        self.bias = Algebra.matrix_sub(self.bias, Algebra.scalar_mult(output_gradient, learning_rate))
        return input_gradient
