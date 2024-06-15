import numpy as np
from convolutional import Convolutional
from helper import Helper

# #### TEST GENERATING KERNELS AND BIASES
# input_shape = (2, 4, 4)  # Example input shape (depth, height, width)
# kernel_size = 2          # Example kernel size
# depth = 3                # Example number of filters

# conv_layer = Convolutional(input_shape, kernel_size, depth)

# print("Initialized Kernels:")
# for d, kernel_set in enumerate(conv_layer.kernels):
#     print(f":::::::::::::::Depth/Filter {d+1}:::::::::::::::")
#     print(kernel_set)
#     # for i, kernel in enumerate(kernel_set):
#     #     print(f" Kernel {iclear}:")k
#     #     for row in kernel:
#     #         print("  ", row)

# print("\nInitialized Biases:")
# for d, bias in enumerate(conv_layer.biases):
#     print(f"Depth {d}:")
#     for row in bias:
#         print(" ", row)



### FOWARDING
input_data = [
    [[1, 2, 0, 1],
     [0, 1, 2, 3],
     [3, 0, 1, 2],
     [2, 3, 0, 1]],
    [[0, 1, 0, 1],
     [2, 2, 2, 2],
     [3, 3, 3, 3],
     [1, 1, 1, 1]]
]

input_shape = (2, 4, 4)  # Example input shape (depth, height, width)
kernel_size = 3 # Example kernel size
depth = 3 # Example number of filters


conv_layer = Convolutional(input_shape, kernel_size, depth, mode='valid')
#Fowarding
print(":::::::::::::::::Forwarding:::::::::::::::::")
output = conv_layer.forward(input_data)
for layer in output:
    print("Matrix")
    for row in layer:
        print(row)

### Backwarding
output_gradient = [
    Helper.create_random_matrix(len(output[0]), len(output[0][0]))
    for _ in range(len(output))
]
print(":::::::::::::::::EXAMPLE: Output gradient:::::::::::::::::")
for layer in output_gradient:
    print("Matrix")
    for row in layer:
        print(row)
print(":::::::::::::::::Backwarding:::::::::::::::::")
input_backprop = conv_layer.backward(output_gradient, learning_rate=0.0001)
for layer in input_backprop:
    print("Matrix")
    for row in layer:
        print(row)

#########################################################################################################
#### Reshape
# from reshape import Reshape
# input_shape = (2, 3, 4)  # Example input shape (depth, height, width)
# output_shape = (24, 1)  # Example output shape (height, width)

# reshape_layer = Reshape(input_shape, output_shape)

# input_data = [
#     [
#         [1.2, 2, 3, 4],
#         [5, 6, 7, 8],
#         [9, 10, 11, 12]
#     ],
#     [
#         [13, 14, 15, 16],
#         [17, 18, 19, 20],
#         [21, 22, 23, 24]
#     ]
# ]

# forward_output = reshape_layer.forward(input_data)
# backward_output = reshape_layer.backward(forward_output, learning_rate=0)  # learning_rate is not used

# print(f"::::::::::::::RESHAPE::::::::::::::")
# print("Input Data:")
# print(input_data)
# print("\nForward Output:")
# print(forward_output)
# print("\nBackward Output:")
# print(backward_output)





#########################################################################################################           
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#### CACULATING
# input_data = [
#     [
#         [1, 6, 2],
#         [5, 3, 1],
#         [7, 0, 4]
#     ]
# ]

# kernel = [
#     [
#         [1, 2],
#         [-1, 0]
#     ]
# ]

# class CustomConvolutional(Convolutional):
#     def __init__(self, input_shape, kernel_size, depth, custom_kernels):
#         super().__init__(input_shape, kernel_size, depth)
#         self.kernels = custom_kernels
        
# input_shape = (1, 3, 3) 
# kernel_size = 2
# depth = 1

# conv_layer = CustomConvolutional(input_shape, kernel_size, depth, [kernel])
# output = conv_layer.forward(input_data)
# print("Output:")
# for layer in output:
#     for row in layer:
#         print(row)