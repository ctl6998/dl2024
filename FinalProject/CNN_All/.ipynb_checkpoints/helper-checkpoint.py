import random

class Helper:
    @staticmethod
    #correlated2d receive: input, kernel, result (matrix form)
    def correlate2d(input, kernel, result, mode='valid'):
        input_height = len(input)
        input_width = len(input) #len(input[0])
        kernel_height = len(kernel)
        kernel_width = len(kernel[0])

        if mode == 'valid':  # No padding
            result_height = input_height - kernel_height + 1
            result_width = input_width - kernel_width + 1
            padded_input = input
        elif mode == 'same':  # Padding to ensure output size matches input size
            result_height = input_height
            result_width = input_width
            pad_height = kernel_height // 2
            pad_width = kernel_width // 2
            padded_input = Helper.pad_with_zeros(input, pad_height, pad_width)
        else:
            raise ValueError(f"What is mode '{mode}'? Supported modes are 'valid', 'same' or 'full' if I'm ok.")

        # Perform convolution
        for i in range(result_height):
            for j in range(result_width):
                for m in range(kernel_height):
                    for n in range(kernel_width):
                        result[i][j] += padded_input[i + m][j + n] * kernel[m][n]
        return result
    
    @staticmethod
    #convolve2d receive: input, kernel, result (matrix form) 
    def convolve2d(input, kernel, result, mode='full'):
        input_height = len(input)
        input_width = len(input) #len(input[0])
        kernel_height = len(kernel)
        kernel_width = len(kernel[0])

        if mode == 'full':  # Full convolution
            result_height = input_height + kernel_height - 1
            result_width = input_width + kernel_width - 1
            padded_input = Helper.pad_with_zeros(input, kernel_height - 1, kernel_width - 1)
        # elif mode == 'same':  # Padding to ensure output size matches input size
        #     result_height = input_height
        #     result_width = input_width
        #     pad_height = kernel_height // 2
        #     pad_width = kernel_width // 2
        #     padded_input = Helper.pad_with_zeros(input, pad_height, pad_width)
        # elif mode == 'valid':  # No padding
        #     result_height = input_height - kernel_height + 1
        #     result_width = input_width - kernel_width + 1
        #     padded_input = input
        else:
            raise ValueError(f"What is mode '{mode}'?")

        # Perform convolution
        for i in range(result_height):
            for j in range(result_width):
                for m in range(kernel_height):
                    for n in range(kernel_width):
                        result[i][j] += padded_input[i + m][j + n] * kernel[m][n]
        return result
    
    @staticmethod
    def rotate180(matrix):
        return [row[::-1] for row in matrix[::-1]]

    @staticmethod
    def pad_with_zeros(input, pad_height, pad_width):
        input_height = len(input)
        input_width = len(input[0])
        padded_height = input_height + 2 * pad_height
        padded_width = input_width + 2 * pad_width

        padded_input = [[0] * padded_width for _ in range(padded_height)]
        for i in range(input_height):
            for j in range(input_width):
                padded_input[i + pad_height][j + pad_width] = input[i][j]
        return padded_input
    
    @staticmethod
    def add_biases(result, biases):
        for i in range(len(result)):
            for j in range(len(result[0])):
                result[i][j] += biases[i][j]
        return result
    
    @staticmethod
    def create_zero_matrix(height, width):
        return [[0 for _ in range(width)] for _ in range(height)]
    
    @staticmethod
    def create_random_matrix(height, width):
        return [[random.uniform(-1, 1) for _ in range(width)] for _ in range(height)]
