import math
### Activation flow
# class Activation:
#     def __init__(self, activation_func, activation_prime):
#         self.activation_func = activation_func
#         self.activation_prime = activation_prime

# class Tanh(Activation):
#     def __init__(self):
#         def tanh(x):
#             return math.tanh(x[0])  # x is a list containing a single element

#         def tanh_prime(x):
#             return 1 - math.tanh(x[0]) ** 2

#         super().__init__(tanh, tanh_prime)

# class Sigmoid(Activation):
#     def __init__(self):
#         def sigmoid(x):
#             return 1 / (1 + math.exp(-x[0]))  # x is a list containing a single element

#         def sigmoid_prime(x):
#             s = sigmoid(x)
#             return s * (1 - s)

#         super().__init__(sigmoid, sigmoid_prime)

# def apply_function(func, matrix):
#     return [func(element) for element in matrix]

# # Example usage:
# tanh_activation = Tanh()
# sigmoid_activation = Sigmoid()

# # Example of using the apply_function:
# input_vector = [[1], [1], [1], [1]]  # 1D matrix (list of lists)

# output_tanh = apply_function(tanh_activation.activation_func, input_vector)
# output_sigmoid = apply_function(sigmoid_activation.activation_func, input_vector)

# print("Tanh activation:")
# print(output_tanh)

# print("\nSigmoid activation:")
# print(output_sigmoid)

### Matrix muplity
from algebra_helper import Algebra
import random
A = [
    [-1.5, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
    
B = [
    [9.3, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
]
    
print("Matrix A:")
for row in A:
    print(row)
    
print("\nMatrix B:")
for row in B:
    print(row)
    
# Perform elementwise multiplication
C = Algebra.elementwise_mult(A, B)
    
print("\nElementwise multiplication of A and B (Matrix C):")
for row in C:
    print(row)
    
print(1.5*2)