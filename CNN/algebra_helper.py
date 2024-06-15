import random

class Algebra:
    @staticmethod
    def matrix_mult(A, B):
        # Matrix multiplication A * B
        result = [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
        return result
    
    @staticmethod
    def matrix_add(A, B):
        # Matrix addition A + B
        result = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
        return result
    
    @staticmethod
    def matrix_sub(A, B):
        # Matrix subtraction A - B
        result = [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
        return result
    
    @staticmethod
    def scalar_mult(A, scalar):
        # Scalar multiplication scalar * A
        result = [[A[i][j] * scalar for j in range(len(A[0]))] for i in range(len(A))]
        return result
    
    @staticmethod
    def transpose(A):
        # Transpose of matrix A
        result = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
        return result
    
    # @staticmethod
    # def elementwise_mult(A, B):
    #     # Elementwise multiplcation (Hadamand pro
    #     return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    
    # Applt for both tensor/3D Matrix (depth, maitrix(height, width)) and 2D Matrix
    @staticmethod
    def elementwise_mult(A, B):
        if isinstance(A[0][0], (int, float)):  # Check if A and B are normal matrices
            # Elementwise multiplication for normal matrices
            if len(A) != len(B) or len(A[0]) != len(B[0]):
                raise ValueError("Matrices must have the same dimensions for elementwise multiplication.")
            return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
        elif isinstance(A[0][0], list):  # Check if A and B are tensors (3D matrices)
            # Elementwise multiplication for tensors
            if len(A) != len(B) or len(A[0]) != len(B[0]) or len(A[0][0]) != len(B[0][0]):
                raise ValueError("Tensors must have the same dimensions for elementwise multiplication.")
            return [[[A[i][j][k] * B[i][j][k] for k in range(len(A[0][0]))] for j in range(len(A[0]))] for i in range(len(A))]
        else:
            raise ValueError("Unsupported input types. A and B should either be normal matrices or tensors.")

    # @staticmethod
    # def apply_function(func, matrix):
    #     return [[func(matrix[i][j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    @staticmethod
    def apply_function(func, matrix):
        if isinstance(matrix[0][0], list):  # Check if the matrix has depth
            return [[[func(matrix[d][i][j]) for j in range(len(matrix[0][0]))] for i in range(len(matrix[0]))] for d in range(len(matrix))]
        else:
            return [[func(matrix[i][j]) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    
    @staticmethod
    def print_3d_matrix(matrix):
        depth = len(matrix)
        height = len(matrix[0]) 
        width = len(matrix[0][0])
        print("::::::::::::::::PRINT MATRIX::::::::::::::::")
        print(f"Depth: {depth}, Height: {height}, Width: {width}")
        print("[")
        for d in range(len(matrix)):
            print("    [")
            for h in range(len(matrix[d])):
                row = ', '.join(f"{value:.2f}" for value in matrix[d][h])
                print(f"        [{row}],")
            if d < len(matrix) - 1:
                print("    ],")
            else:
                print("    ]")
        print("]")
    