import random
import math

class NeuralNetwork:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.weights = []
        self.biases = []

    def init_weights(self, init_type='random', file_path=None):
        if init_type == 'random':
            self.weights = [[random.random() for _ in range(self.layers[i])] for i in range(1, self.num_layers)]
        elif init_type == 'file' and file_path:
            with open(file_path, 'r') as file:
                for line in file:
                    weights_row = [float(val) for val in line.strip().split()]
                    self.weights.append(weights_row)

    def init_biases(self, init_type='random', file_path=None):
        if init_type == 'random':
            self.biases = [[random.random() for _ in range(layer)] for layer in self.layers[1:]]
        elif init_type == 'file' and file_path:
            with open(file_path, 'r') as file:
                for line in file:
                    biases_row = [float(val) for val in line.strip().split()]
                    self.biases.append(biases_row)

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def feedforward(self, input_data):
        a = input_data
        for w, b in zip(self.weights, self.biases):
            z = [sum(w[j] * a_j for a_j in a) + b_i for j, b_i in enumerate(b)]
            a = [self.sigmoid(z_i) for z_i in z]
        return a

if __name__ == "__main__":
    with open('./config.txt', 'r') as file:
        num_layers = int(file.readline().strip())
        layers = [int(file.readline().strip()) for _ in range(num_layers)]
        
    network = NeuralNetwork(layers)
    network.init_weights(init_type='file', file_path='./weights.txt')
    network.init_biases(init_type='file', file_path='./biases.txt')
    x = [[0, 0], [0, 1], [1, 0], [1, 1]] 
    print("Output of XOR gate from file:")
    for input_data in x:
        output = network.feedforward(input_data)
        print(f"Input: {input_data}, Output: {output}")
        
    network_2 = NeuralNetwork(layers)
    network_2.init_weights(init_type='random')
    network_2.init_biases(init_type='random')
    x = [[0, 0], [0, 1], [1, 0], [1, 1]] 
    print("Output of XOR gate from random:")
    for input_data in x:
        output = network_2.feedforward(input_data)
        print(f"Input: {input_data}, Output: {output}")
        
        
