from softmax import Softmax

def main():
    input_data = [
        [-7.04],
        [0.40]
    ]
    
    softmax_layer = Softmax()
    
    # Forward
    softmax_output = softmax_layer.forward(input_data)
    print("Softmax output:")
    for row in softmax_output:
        print(row)
    
    # Backward 
    input_gradient = softmax_layer.backward(softmax_output, learning_rate=0.1)
    print("\nInput gradient:")
    for row in input_gradient:
        print(row)

if __name__ == "__main__":
    main()
