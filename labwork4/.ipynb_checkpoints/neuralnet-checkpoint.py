import math
import random

#Activate function declaration
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

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
    
#For bias
class BiasNeuron(Base):
    def __init__(self):
        pass
    

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
    def __init__(self, nodes, activation=sigmoid):
        self.id = f"{self.__class__.__name__}{Layer.id}"
        Layer.id += 1
        self.bias = 1
        self.neurons = [Neuron(activation) for _ in range(nodes)]
        
    def describe(self):
        self.log(f"{len(self.neurons)} neurons")
        for neuron in self.neurons:
            neuron.describe()
            
    def forward(self, layerLinks, inputs):
        layerLinks = [layerLink for layerLink in layerLinks if layerLink.toLayer == self]
        layerLink = layerLinks[0]
        self.log(f"Foward: a total of {len(layerLinks)} layer links, first is {layerLink}")
        prevLayer = layerLink.fromLayer
        self.log(f"Foward: previous layer is {prevLayer}")
        for neuron in self.neurons:
            weights = []
            outputs = []
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
        self.links = [] #link = tuple(fromNeuron, toNeun, weight)
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
    def __init__(self, config="config.txt"):
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
        self.log("======Reading config file======")
        with open(config, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line and not line.startswith("#")]
        
        # Init layers
        n = int(lines[0])
        for i in range(1, n + 1):
            nodes = int(lines[i])
            self.layers.append(Layer(nodes))
                            
        # Init links
        for i in range(1, len(self.layers)):
            lastLayer = self.layers[i - 1]
            nextLayer = self.layers[i]
            self.layerLinks.append(LayerLink(lastLayer, nextLayer))
            
        # Init weights
        for i, layerLink in enumerate(self.layerLinks):
            weightStr = lines[n + 1 + i].strip().split(",")
            weights = [float(w.strip()) for w in weightStr]
            if len(weights) != len(layerLink.links):
                raise ValueError(f"LayerLink {layerLink.id} does not match with weights in line {n+1+i}")
            else:
                for j, link in enumerate(layerLink.links):
                    link.weight = weights[j]

            
        self.log("======FINISHED CONFIGURATION======")
                            
    def forward(self, inputs):
        self.log("======FORWARDING=======")
        if len(inputs) != len(self.layers[0].neurons):
            raise ValueError("Number of input neurons does not match the size of input data")
        for i in range(len(inputs)):
            self.layers[0].neurons[i].output = inputs[i]
        for layers in self.layers[1:]:
            layers.forward(self.layerLinks, inputs)

if __name__ == "__main__":
    network = Network(config="config.txt")
    network.describe()
    
    # Input data
    input_set = [
        [0, 0], 
        [0, 1], 
        [1, 0], 
        [1, 1] 
    ]
    
    for inputs in input_set:
        print("Running network on inputs:")
        print("Input:", inputs)
        network.forward(inputs)
        output_layer = network.layers[-1]
        output_values = [neuron.output for neuron in output_layer.neurons]
        print("Output values:", output_values)
        print(f"================================================")
