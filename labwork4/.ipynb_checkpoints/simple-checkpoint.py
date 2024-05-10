import random
import math

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        # Initialize weights and biases randomly
        self.weights = [[random.random() for _ in range(layer_sizes[i-1])] for i in range(1, self.num_layers)]
        self.biases = [[random.random() for _ in range(layer_sizes[i])] for i in range(1, self.num_layers)]
        
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + math.exp(-z))
    
    def feedforward(self, x):
        a = x
        for w, b in zip(self.weights, self.biases):
            z = [sum(w[j] * a[j] for j in range(len(a))) + b_i for b_i in b]
            a = [self.sigmoid(z_i) for z_i in z]
        return a

    def load_weights_and_biases(self, weights_file, biases_file):
        pass

if __name__ == "__main__":
    layer_sizes = [2, 2, 1]  # 2 input neurons, 2 hidden neurons, 1 output neuron
    nn = NeuralNetwork(layer_sizes)
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Input data
    print("Output of XOR gate:")
    for input_data in x:
        output = nn.feedforward(input_data)
        print(f"Input: {input_data}, Output: {output}")
