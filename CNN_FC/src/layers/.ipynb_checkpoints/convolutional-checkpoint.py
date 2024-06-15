import random
from layers.layer import Layer
from helpers.convolutional_helper import Helper

# input_shape = (2, 4, 4)  # Example input shape (depth, height, width)
# kernel_size = 2          # Example kernel size
# depth = 3                # Example number of filters
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth, mode='valid'):
        input_depth, input_height, input_width = input_shape
        self.depth = depth #Kernel depth or number of filter
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.mode = mode
        # Calculate shape of output based on mode
        if mode == 'valid':
            self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        elif mode == 'same':
            self.output_shape = (depth, input_height, input_width)
        else:
            raise ValueError(f"Unsupported mode '{mode}'")
        # Compute shape of kernel
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        # Generate kernels & biases
        self.kernels = self.initialize_kernels(*self.kernels_shape)
        self.biases = self.initialize_biases(*self.output_shape)
        
        
    def initialize_kernels(self, depth, input_depth, kernel_height, kernel_width):
        return [
            [
                Helper.create_random_matrix(kernel_height, kernel_width)
                for _ in range(input_depth)
            ]
            for _ in range(depth)
        ]

    def initialize_biases(self, depth, output_height, output_width):
        return [
            Helper.create_random_matrix(output_height, output_width)
            for _ in range(depth)
        ]
    
    ################################## FOWARDING ################################## 
    ### The foward using crosss-correlation to process
    ### Cross-correlation come with input, kernel, output shape, mode (representing padding). 
    ### Cross-correlation stride is always 1 (assuming)
    def forward(self, input):
        super().forward(input)
        self.input = input
        # Initialize output with zeros
        self.output = [
            [
                [0 for _ in range(self.output_shape[2])] 
                for _ in range(self.output_shape[1])
            ] 
            for _ in range(self.depth)
        ]
        # Perform convolution and add biases
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] = Helper.correlate2d(self.input[j], self.kernels[i][j], self.output[i], self.mode)
            self.output[i] = Helper.add_biases(self.output[i], self.biases[i])
        return self.output
    
    ################################## BACKPROPAGATION ################################## 
    ### The backward using Gradient Descent with learning rate to update kernels and biases
    ### Cross-correlation stride is always 1 (assuming)
    ### Convolution is actually cross-correlation with 180 rotate of kernel - see math explanation
    ### Remind: kernel size must odd
    def backward(self, output_gradient, learning_rate):
        super().backward(output_gradient, learning_rate)
        # Initialize kernel gradient and input graident matix (should be all 0)
        kernels_gradient = [
            [
                Helper.create_zero_matrix(len(self.kernels[0][0]), len(self.kernels[0][0][0]))
                for _ in range(self.input_depth)
            ]
            for _ in range(self.depth)
        ]
        input_gradient = [
            [
                [0 for _ in range(self.input_shape[2])]
                for _ in range(self.input_shape[1])
            ]
            for _ in range(self.input_depth)
        ]
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i][j] = Helper.correlate2d(self.input[j], output_gradient[i], kernels_gradient[i][j], self.mode) #Remind: kernel size must odd
                input_gradient[j] = Helper.convolve2d(output_gradient[i], Helper.rotate180(self.kernels[i][j]), input_gradient[j], mode='full')

        for i in range(self.depth):
            for j in range(self.input_depth):
                for m in range(len(self.kernels[i][j])):
                    for n in range(len(self.kernels[i][j][0])):
                        self.kernels[i][j][m][n] -= learning_rate * kernels_gradient[i][j][m][n]

        for i in range(self.depth):
            for m in range(len(self.biases[i])):
                for n in range(len(self.biases[i][0])):
                    self.biases[i][m][n] -= learning_rate * output_gradient[i][m][n]

        return input_gradient