class Layer:
    def __init__(self,log=False):
        self.input = None
        self.output = None
        self.log = log

    def forward(self, input):
        # print(f"::::::::::::::::FWD-[{self.__class__.__name__.upper()}_LAYER]::::::::::::::::")
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # print(f"::::::::::::::::BWD-[{self.__class__.__name__.upper()}_LAYER]::::::::::::::::")
        # TODO: update parameters and return input gradient
        pass