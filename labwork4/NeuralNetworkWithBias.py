import math
import random

#Activate function declaration
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Base:
    def __init__(self):
        pass
    
    def describe(self):
        pass
    
    def log(self, message):
        print(f"{self.id}: {message}")
    
    def __str__(self):
        return self.id
    
class Neuron(Base):
    id = 0
    def __init__(self, activation=sigmoid):
        self.id = f'{self.__class__.__name__}{Neuron.id}'
        Neuron.id += 1
        self.activation = activation
        self.output = 0
    
    def describe(self):
        self.log(f'Activation {self.activation.__name__}')
        
    def linearSum(self, weights, inputs):
        return sum((weights[i] * inputs[i] for i in range(len(weights))))
        
    def activate(self, z):
        return self.activation(z)
    
    def forward(self):
        return self.output

class BiasNeuron(Base):
    id = 0
    def __init__(self):
        self.id = f"{self.__class__.__name__}{BiasNeuron.id}"
        BiasNeuron.id += 1
        self.output = 1    
    
class Link(Base):
    id = 0
    def __init__(self, fromNeuron, toNeuron, weight=0):
        self.id = f"{self.__class__.__name__}{Link.id}"
        Link.id += 1
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron
        self.weight = weight
            
    
class Layer(Base):
    id = 0
    def __init__(self, nodes, activation=sigmoid, is_output_layer=False):
        self.id = f"{self.__class__.__name__}{Layer.id}"
        Layer.id += 1
        self.neurons = [Neuron(activation) for _ in range(nodes)]
        if not is_output_layer:
            self.bias_neuron = BiasNeuron() 
            
    def describe(self):
        if hasattr(self, 'bias_neuron'):
            self.log(f"{len(self.neurons)} neurons + 1 bias neuron")
            for neuron in self.neurons:
                neuron.describe()
        else:
            self.log(f"{len(self.neurons)} neurons")
            for neuron in self.neurons:
                neuron.describe()
    
    def forward(self, layerLinks, inputs):
        # Append bias neuron output to inputs if bias neuron exists
        if hasattr(self, 'bias_neuron'):
            inputs_with_bias = [self.bias_neuron.output] + inputs
            # print("Input with bias:", self.bias_neuron.output)
        else:
            inputs_with_bias = inputs
        layerLinks = [layerLink for layerLink in layerLinks if layerLink.toLayer == self]
        layerLink = layerLinks[0]
        # self.log(f"Foward: a total of {len(layerLinks)} layer links, first is {layerLink}")
        prevLayer = layerLink.fromLayer
        # self.log(f"Foward: connecting from {prevLayer}")
    
        # Print connections from input neurons to neurons in this layer
        for neuron in self.neurons:
            weights = []
            outputs = []
            if hasattr(self, 'bias_neuron'):
                for link in layerLink.links:
                    if link.toNeuron == neuron:
                        weights.append(link.weight)
                        if link.fromNeuron == self.bias_neuron:
                            # If the connection is from the bias neuron, use its output
                            outputs.append(self.bias_neuron.output)
                        else:
                            outputs.append(link.fromNeuron.output)
                        self.log(f"{link.fromNeuron.id} -> {neuron.id}: (weight: {link.weight:.2f})")
            else:
                for link in layerLink.links:
                    if link.toNeuron == neuron:
                        weights.append(link.weight)
                        outputs.append(link.fromNeuron.output)
                        self.log(f"{link.fromNeuron.id} -> {neuron.id}: (weight: {link.weight:.2f})")
            z = neuron.linearSum(weights, outputs)
            neuron.output = neuron.activate(z)
            neuron.log(f"{neuron.id} output: z={z:.2f}, a={neuron.output:.2f}")

    
        return [neuron.output for neuron in self.neurons]

                    
        
class LayerLink(Base):
    id = 0
    def __init__(self, fromLayer, toLayer, weights=[]):
        self.id = f"{self.__class__.__name__}{LayerLink.id}"
        LayerLink.id += 1
        self.fromLayer = fromLayer
        self.toLayer = toLayer
        self.links = [] #link = (fromNeuron, toNeuron, weight)
        if hasattr(fromLayer, 'bias_neuron'):
            for j in range(len(toLayer.neurons)):
                link = Link(fromLayer.bias_neuron, toLayer.neurons[j], random.random()) #randomize weight
                self.links.append(link)
        for i in range(len(fromLayer.neurons)):
            for j in range(len(toLayer.neurons)):
                link = Link(fromLayer.neurons[i], toLayer.neurons[j], random.random()) #randomize weight
                self.links.append(link)
    
    def describe(self):
        self.log(f"{len(self.links)} links from {self.fromLayer.id} to {self.toLayer.id}")
        for link in self.links:
            self.log(f"{link.fromNeuron.id} -> {link.toNeuron.id} ({link.weight:.2f})")
            
            
class Network(Base):
    id = 0
    def __init__(self, config):
        self.id = f"{self.__class__.__name__}{Network.id}"
        Network.id += 1
        self.layers = []
        self.layerLinks = []
        self.parseConfig(config)
    
    def describe(self):
        self.log(f"Start describing network {self.id}")
        self.log(f"{self.id}: {len(self.layers)}: layers")
        for layer in self.layers:
            layer.describe()
                            
    def parseConfig(self, config):
        self.log("======PARSING CONFIGURATION======")
        with open(config, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line and not line.startswith("#")]
        
        # Init layers
        n = int(lines[0])
        for i in range(1, n + 1):
            nodes = int(lines[i])
            if i == n:
                self.layers.append(Layer(nodes, is_output_layer=True))
            else:
                self.layers.append(Layer(nodes))
        # print(f"Number of layer from config:", len(self.layers))
                            
        # Init links
        for i in range(1, len(self.layers)):
            lastLayer = self.layers[i - 1]
            nextLayer = self.layers[i]
            self.layerLinks.append(LayerLink(lastLayer, nextLayer))
            
        # Init weights
        for i, layerLink in enumerate(self.layerLinks):
            weightStr = lines[n + 1 + i].strip().split(",")
            weights = [float(w.strip()) for w in weightStr]
            # print(f"Weights from config:", weights)
            # print(f"Layer links:", len(layerLink.links))
            if len(weights) != len(layerLink.links):
                raise ValueError(f"LayerLink {layerLink.id} does not match with weights in line {n+1+i}")
            else:
                for j, link in enumerate(layerLink.links):
                    link.weight = weights[j]

        self.log("======FINISHED CONFIGURATION======")
                            
    def forward(self, inputs):
        if len(inputs) != len(self.layers[0].neurons):
            raise ValueError("Number of input neurons does not match the size of input data")
        for i in range(len(inputs)):
            self.layers[0].neurons[i].output = inputs[i]
        for layers in self.layers[1:]:
            layers.forward(self.layerLinks, inputs)

    def backward(self, targets, learning_rate):
        if len(targets) != len(self.layers[-1].neurons):
            raise ValueError("Number of target values does not match the size of output layer")

        # Compute output layer gradients
        output_derivatives = [sigmoid_derivative(neuron.output) for neuron in self.layers[-1].neurons]
        output_deltas = [(
                # Loss compare to give target
                targets[i] - self.layers[-1].neurons[i].output) * output_derivatives[i] 
                for i in range(len(self.layers[-1].neurons)
        )]

        # Compute gradients and deltas for hidden layers
        hidden_deltas = [output_deltas]
        # Derivatives can be used for threshold
        # hidden_derivatives = [output_derivatives]

        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            layer_deltas = []
            layer_derivatives = []

            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                derivative = sigmoid_derivative(neuron.output)
                layer_derivatives.append(derivative)

                error = sum(
                    hidden_deltas[-1][k] * next((link.weight for link in self.layerLinks[i].links if link.fromNeuron == neuron and link.toNeuron == next_layer.neurons[k]), 0)
                    for k in range(len(next_layer.neurons))
                )
                # delta_l = (W_(l+1)^T * delta_(l+1)) * f'(z_l)
                layer_deltas.append(error * derivative)

            hidden_deltas.append(layer_deltas)
            # hidden_derivatives.append(layer_derivatives)

        # Update weights
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]

            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                delta = hidden_deltas[-1][j]

                for k in range(len(prev_layer.neurons)):
                    prev_neuron = prev_layer.neurons[k]
                    link = next((link for link in self.layerLinks[i - 1].links if link.fromNeuron == prev_neuron and link.toNeuron == neuron), None)
                    old_weight = link.weight
                    # W_l = W_l - learning_rate * gradient = W_l - learning_rate * delta * prev_neuron.output
                    link.weight += learning_rate * delta * prev_neuron.output
                    self.log(f"Updated weight: {prev_neuron.id} -> {neuron.id}: {old_weight:.4f} -> {link.weight:.4f}")

                # Update bias weight
                bias_link = next((link for link in self.layerLinks[i - 1].links if link.fromNeuron == prev_layer.bias_neuron and link.toNeuron == neuron), None)
                old_bias_weight = bias_link.weight
                bias_link.weight += learning_rate * delta
                self.log(f"Updated bias weight: {prev_layer.bias_neuron.id} -> {neuron.id}: {old_bias_weight:.4f} -> {bias_link.weight:.4f}")

    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs + 1):
            print(f"====================RUNNING EPOCH {i}====================")
            for inputs, targets in zip(input_set, target_set):
                print(f"=======================================================")
                print(f"Check with input: {inputs} for target: {targets}")
                self.log(f"======FORWARDING FOR INPUT {inputs}=======")
                self.forward(inputs)
                self.log(f"======BACKPROPAGATION FOR INPUT {inputs}=======")
                self.backward(targets, learning_rate)

if __name__ == "__main__":
    network = Network(config="config-bias.txt")
    network.describe()
    
    input_set = [[0, 0], [0, 1], [1, 0], [1, 1]]
    target_set = [[0], [1], [1], [0]]
    
    epochs = 1000
    learning_rate = 0.1
    network.train(input_set, target_set, epochs, learning_rate)
