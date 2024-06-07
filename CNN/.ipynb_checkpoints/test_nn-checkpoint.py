¬from dense import Dense
from activation import Activation
from activations import Sigmoid, Tanh

# dense(10x1) -> activation -> dense(2x1)

# Initialize the dense layer with input size 10 and output size 2
dense_layer = Dense(input_size=10, output_size=4)

function = Sigmoid()
    
# Initialize the activation layer with the sigmoid function
activation_layer = Activation(activation=function.activation, activation_derivative=function.activation_derivative)
    
# Input data (10x1 matrix)
input_data = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    
# Forward pass through the dense layer
dense_output = dense_layer.forward(input_data)
print("Dense layer output:", dense_output)
    
# Forward pass through the activation layer
activation_output = activation_layer.forward(dense_output)
print("Activation forward output:", activation_output)
    
# Output gradient (4x1 matrix)
output_gradient = [[1], [1], [1], [1]]
# print(type(output_gradient[]))
print("::::::::::::::::::::::")

# Backward pass through the activation layer
activation_backward_output = activation_layer.backward(output_gradient, learning_rate=0.01)
print("Activation backward output:", activation_backward_output)
    
# Backward pass through the dense layer
dense_layer.backward(activation_backward_output, learning_rate=0.01)

    
# Print updated weights and biases
print("Updated dense layer weights:", dense_layer.weights)
print("Updated dense layer bias:", dense_layer.bias)
