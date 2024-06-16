import math
from layer import Layer

class Softmax(Layer):
    ################################## FOWARDING ################################## 
    ### The foward of Softmax layer
    ### Softmax convert dense layer [x1,x2,...,xi,...,xn] to [y1,y2,...yi,...,yn]
    ### With yi = exp(xi) / sum(exp(xi))
    # def forward(self, input):
    #     super().forward(input)
    #     self.input = input
    #     exps = [math.exp(x[0]) for x in input]
    #     sum_exps = sum(exps)
    #     self.output = [[math.exp(x[0]) / sum_exps] for x in input]
    #     if self.log==True:
    #         print(f"Softmax forward input:")
    #         Algebra.print_3d_matrix(self.input)
    #         print(f"Softmax forward output:")
    #         Algebra.print_3d_matrix(self.output)
    #     return self.output
    def forward(self, input):
        exp_input = [math.exp(i[0]) for i in input]
        sum_exp_input = sum(exp_input)
        self.output = [i / sum_exp_input for i in exp_input]
        #print(self.output)
        return self.output
    
    
    ################################## BACKPROPAGATION ################################## 
    ### The back propagation of Softmax layer
    ### Softmax convert gradient output dE/dY to dE/dX, assume X, Y is dense layer with n neurons
    ### dE/dX = [M .(Element-wise Multiplcation). (I - M-transposed)] .(matrix-multiplication) . dE/dY
    ### For easy interpretation: 
    ### dE/dX = M*(I-Mt).dE/dY = J.dE/dY
    def backward(self, output_gradient, learning_rate):
        n = len(self.output)
        # Identity matrix I
        identity_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        # Caculating J
        J = [
            [self.output[i] * (identity_matrix[i][j] - self.output[j]) for j in range(n)]
            for i in range(n)
        ]
        input_gradient = [sum(J[i][j] * output_gradient[j] for j in range(n)) for i in range(n)]
        
        return input_gradient


