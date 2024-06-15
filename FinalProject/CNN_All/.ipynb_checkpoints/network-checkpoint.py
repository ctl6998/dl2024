def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_derivative, x_train, y_train, epochs = 1000, learning_rate = 0.01, print_progress = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            # print(f"::::::::::::::::EPOCH {e} FORWARDING::::::::::::::::")
            output = predict(network, x)
            # print(f"Foward output:",output)

            # error
            # print(f"::::::::::::::::CACULATING LOSS::::::::::::::::")
            # # print(f"Initial input",x)
            # print(f"Real (Encoded label)",y)
            # # print(f"Predict",output)
            error += loss(y, output)

            # backward
            # print(f"::::::::::::::::EPOCH {e} BACKPROPAGATION::::::::::::::::")
            grad = loss_derivative(y, output)
            # print(f"Gradient output:", grad)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if print_progress:
            print(f"Epoch: {e + 1}/{epochs}, error={error}")