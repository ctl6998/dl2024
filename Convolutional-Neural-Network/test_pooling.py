from pooling import MaxPooling

# Create an instance of MaxPoolingLayer
max_pool_layer = MaxPooling(pool_size=(2, 2), stride=2, log=True)

# Example input data (3D)
input_data_3d = [
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]],
    [[17, 18, 19, 20],
     [21, 22, 23, 24],
     [25, 26, 27, 32],
     [29, 30, 31, 28]]
]
print("FW:::::::::::::::::::::::::::::::")
print("Test on input tensor")
output_data_3d = max_pool_layer.forward(input_data_3d)
print("Input shape (3D):", max_pool_layer.input_shape())
print("Output shape (3D):", max_pool_layer.output_shape())
print("Output (3D):\n", output_data_3d)

# Example input data (2D)
# input_data_2d = [
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 16, 12],
#     [13, 14, 15, 2]
# ]
# print("FW:::::::::::::::::::::::::::::::")
# print("Test on input not tensor")
# output_data_2d = max_pool_layer.forward(input_data_2d)
# print("Input shape (2D):", max_pool_layer.input_shape())
# print("Output shape (2D):", max_pool_layer.output_shape())
# print("Output (2D):\n", output_data_2d)

# Example output gradient (use the same shape as output_data_3d or output_data_2d)
output_gradient = [
    [[1, 2],
     [3, 4]],
    [[5, 6],
     [7, 8]]
]
print("BW:::::::::::::::::::::::::::::::")
input_gradient = max_pool_layer.backward(output_data_3d, learning_rate=0.1)
print("\nInput gradient shape:", max_pool_layer.input_shape())
print("Input gradient:\n", input_gradient)