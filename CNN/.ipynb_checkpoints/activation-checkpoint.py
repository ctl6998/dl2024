import numpy as np
from layer import Layer
from algebra_helper import Algebra

class Activation(Layer):
    ################################## INITIALIZE A.F. AND DERIVATIVES OF A.F. ################################## 
    def __init__(self, activation, activation_derivative, log=False):
        self.activation = activation
        self.activation_derivative = activation_derivative
        super().__init__(log)
    
    ################################## FOWARDING ##################################
    ### Forward: Y=f(X), input matrix size equal output matrix size
    def forward(self, input):
        super().forward(input)
        self.input = input
        # print(input)
        # Algebra.print_3d_matrix(input)
        if self.log==True:
            print(f"Activation ouput:")
            Algebra.apply_function(self.activation, self.input)
        return Algebra.apply_function(self.activation, self.input)

    ################################## BACKWARD ##################################
    ### Backward return loss gradient dE/dX and use it for previous layer
    ### Return matrix with same size: dE/dX = dE/dY .(Element-wise Multiplcation). f'(X)  
    def backward(self, output_gradient, learning_rate):
        super().backward(output_gradient, learning_rate)
        if self.log==True:
            print(f"Backward in dE/dY:")
            Algebra.print_3d_matrix(output_gradient)
            print(f"Backward in f'(X):")
            Algebra.print_3d_matrix(Algebra.apply_function(self.activation_derivative, self.input))
        return Algebra.elementwise_mult(output_gradient, Algebra.apply_function(self.activation_derivative, self.input))
