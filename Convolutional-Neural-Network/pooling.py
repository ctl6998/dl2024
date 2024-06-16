from layer import Layer

class MaxPooling(Layer):
    def __init__(self, pool_size=(2, 2), stride=(2, 2), log=False):
        self.pool_height, self.pool_width = pool_size
        self.stride_height, self.stride_width = stride, stride
        self.log = log
        self.input = None
        self.output = None
        self.mask = None  # To store the mask of maximum elements for backpropagation

    def forward(self, input_data):
        super().forward(input)
        self.input = input_data
        self.output = []
        
        if isinstance(input_data[0][0], list):  # Check if input_data is tensor/has depth
            # print("yes this is tensor")
            num_channels = len(input_data)
            for channel_data in input_data:
                output_channel = self._max_pool(channel_data)
                self.output.append(output_channel)
        else:
            # print("no this is not tensor")
            num_channels = 1
            self.output = self._max_pool(input_data)

        return self.output

    def backward(self, output_gradient, learning_rate):
        super().backward(output_gradient, learning_rate)
        input_gradient = []

        if isinstance(output_gradient[0][0], list):  # Check if input_data is tensor/has depth
            # print("yes this is tensor")
            for c in range(len(self.input)):
                input_gradient.append(self._backprop_3d(output_gradient[c], c))
        else:
            # print("no this is not tensor")
            input_gradient = self._backprop_2d(output_gradient)

        return input_gradient
    
    def _max_pool(self, input_data):
        output_height = (len(input_data) - self.pool_height) // self.stride_height + 1
        output_width = (len(input_data[0]) - self.pool_width) // self.stride_width + 1

        output = [[0] * output_width for _ in range(output_height)]

        for i in range(output_height):
            for j in range(output_width):
                start_i = i * self.stride_height
                start_j = j * self.stride_width
                pool_region = [row[start_j:start_j + self.pool_width] for row in input_data[start_i:start_i + self.pool_height]]
                output[i][j] = max(max(region) for region in pool_region)

        return output

    def input_shape(self):
        if isinstance(self.input[0][0], list):  # Check if input_data is tensor
            depth, height, width = len(self.input), len(self.input[0]), len(self.input[0][0])
            return depth, height, width
        else:
            height, width = len(self.input), len(self.input[0])
            return height, width

    def output_shape(self):
        if isinstance(self.output[0][0], list):  # Check if output is tensor 
            depth, height, width = len(self.output), len(self.output[0]), len(self.output[0][0])
            return depth, height, width
        else:
            height, width = len(self.output), len(self.output[0])
            return height, width

    def _backprop_3d(self, output_gradient, channel_index):
        input_gradient_channel = [[0] * len(self.input[0][0]) for _ in range(len(self.input[0]))]

        for i in range(len(output_gradient)):
            for j in range(len(output_gradient[0])):
                start_i = i * self.stride_height
                start_j = j * self.stride_width
                pool_region = [row[start_j:start_j + self.pool_width] for row in self.input[channel_index][start_i:start_i + self.pool_height]]
                max_value = max(max(region) for region in pool_region)
                
                for ii in range(self.pool_height):
                    for jj in range(self.pool_width):
                        if pool_region[ii][jj] == max_value:
                            input_gradient_channel[start_i + ii][start_j + jj] += output_gradient[i][j]

        return input_gradient_channel

    def _backprop_2d(self, output_gradient):
        input_gradient = [[0] * len(self.input[0]) for _ in range(len(self.input))]

        for i in range(len(output_gradient)):
            for j in range(len(output_gradient[0])):
                start_i = i * self.stride_height
                start_j = j * self.stride_width
                pool_region = [row[start_j:start_j + self.pool_width] for row in self.input[start_i:start_i + self.pool_height]]
                max_value = max(max(region) for region in pool_region)
                
                for ii in range(self.pool_height):
                    for jj in range(self.pool_width):
                        if pool_region[ii][jj] == max_value:
                            input_gradient[start_i + ii][start_j + jj] += output_gradient[i][j]

        return input_gradient
