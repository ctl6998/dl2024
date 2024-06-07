from layer import Layer
from algebra_helper import Algebra
    
### Reshape rule: depth*height*width = height*width
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        super().forward(input)
        # Algebra.print_3d_matrix(input)
        return self.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        super().backward(output_gradient, learning_rate)
        return self.reshape(output_gradient, self.input_shape)
    
    ################################## FLATTEN ################################## 
    ### Flaten input shape (depth, height, width) to 1D array
    ### .extend method adds each verified integer item of the list returned by self.flatten(item) to flat
    def flatten(self, array):
        if isinstance(array, (int, float)):
            return [array]
        flat = []
        for item in array:
            flat.extend(self.flatten(item))
        return flat
    
    ################################## CONVERSION ################################## 
    ### Convert 1D array into specific shape (depth=1, height, width)
    def reshape(self, array, shape):
        reshaped = []
        flat = self.flatten(array)
        current_index = 0

        def build_dim(shape):
            nonlocal current_index
            if len(shape) == 1:
                sub_array = flat[current_index:current_index + shape[0]]
                current_index += shape[0]
                return sub_array
            return [build_dim(shape[1:]) for _ in range(shape[0])]

        reshaped = build_dim(shape)
        return reshaped
    
