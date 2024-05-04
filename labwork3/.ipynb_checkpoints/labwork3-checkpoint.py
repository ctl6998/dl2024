from math import log, exp

# DATASET
# DATASET = []
DATASET = [
    (2, -3, 0),
    (7, 7, 1),
    (9, 15, 1),
    (4, 2, 0),
    (1, -3, 0),
    (5, 10, 1),
    (8, 5, 0),
    (3, -1, 0),
    (1, 6, 1),
    (10, 12, 1)
]

# with open('dataset.csv', 'r') as f:
#     next(f)  # Skip the header
#     for line in f:
#         att1, att2, label = map(float, line.strip().split(','))
#         DATASET.append((att1, att2, label))
# print(DATASET)


# Gradient Descent (2d Function): param1 * x1 + param2 * x2 + param0
# L: learning rate
# t: threshold
# f: loss function
def gradientDescend3d(param0, param1, param2, L, t, f, d_w0, d_w1, d_w2):
    w0 = param0
    w1 = param1
    w2 = param2
    iteration = 0
    while True:
        oldF = f(w0, w1, w2)
        w0 = w0 - L * d_w0(w0, w1, w2)
        w1 = w1 - L * d_w1(w0, w1, w2)
        w2 = w2 - L * d_w2(w0, w1, w2)
        newF = f(w0, w1, w2)
        check = abs(newF - oldF)
        print(f"Iteration: {iteration}\t\tw0:{round(w0, 2)}\t\tw1:{round(w1, 2)}\t\tw2:{round(w2, 2)}\t\tLoss change:{round(newF,4)}-{round(oldF,4)}={round(check, 4)}")
        iteration += 1
        if check < t:
            print("Finished")
            return (w0, w1, w2)

# Avoid too small/too large sigmoid value
epsilon = 1e-5
def stable_sigmoid(z):
    if z >= 0:
        return 1 / (1 + exp(-z))
    else:
        return exp(z) / (1 + exp(z))

# Diff Function: diff = loss function, diff_w0 = loss function with gradient chagne to parameter
sigmoid=lambda w0, w1, w2, xi_1, xi_2: stable_sigmoid(w1 * xi_1 + w2 * xi_2 + w0)
diff=lambda w0, w1, w2, xi_1, xi_2, yi: - yi*log(sigmoid(w0, w1, w2, xi_1, xi_2) + epsilon) - (1-yi)*log(1 - sigmoid(w0, w1, w2, xi_1, xi_2) + epsilon)
diff_w0=lambda w0, w1, w2, xi_1, xi_2, yi: -( yi/sigmoid(w0, w1, w2, xi_1, xi_2) + (1-yi)/(1-sigmoid(w0, w1, w2, xi_1, xi_2) + epsilon) ) * 1/( sigmoid(w0, w1, w2, xi_1, xi_2) ** 2 ) * (-1) * exp( -(w1 * xi_1 + w2 * xi_2 + w0) )
diff_w1=lambda w0, w1, w2, xi_1, xi_2, yi: -( yi/sigmoid(w0, w1, w2, xi_1, xi_2) + (1-yi)/(1-sigmoid(w0, w1, w2, xi_1, xi_2) + epsilon) ) * 1/( sigmoid(w0, w1, w2, xi_1, xi_2) ** 2 ) * (-xi_1) * exp( -(w1 * xi_1 + w2 * xi_2 + w0) )
diff_w2=lambda w0, w1, w2, xi_1, xi_2, yi: -( yi/sigmoid(w0, w1, w2, xi_1, xi_2) + (1-yi)/(1-sigmoid(w0, w1, w2, xi_1, xi_2) + epsilon) ) * 1/( sigmoid(w0, w1, w2, xi_1, xi_2) ** 2 ) * (-xi_2) * exp( -(w1 * xi_1 + w2 * xi_2 + w0) )

# Loss Function: just sum and devide
loss = lambda dataset, w0, w1, w2: sum( [diff(w0, w1, w2, data[0], data[1], data[2]) for data in dataset] ) / len(dataset)
loss_w0 = lambda dataset, w0, w1, w2: sum( [diff_w0(w0, w1, w2, data[0], data[1], data[2]) for data in dataset] ) / len(dataset)
loss_w1 = lambda dataset, w0, w1, w2: sum( [diff_w1(w0, w1, w2, data[0], data[1], data[2]) for data in dataset] ) / len(dataset)
loss_w2 = lambda dataset, w0, w1, w2: sum( [diff_w2(w0, w1, w2, data[0], data[1], data[2]) for data in dataset] ) / len(dataset)

# Logistic Regression    
def logisticRegression(dataset):
    w0, w1, w2 = 0, 0.05, 0.005
    f = lambda w0, w1, w2: loss(dataset, w0, w1, w2)
    d_w0 = lambda w0, w1, w2: loss_w0(dataset, w0, w1, w2)
    d_w1 = lambda w0, w1, w2: loss_w1(dataset, w0, w1, w2)
    d_w2 = lambda w0, w1, w2: loss_w2(dataset, w0, w1, w2)
    (w0, w1, w2) = gradientDescend2d(w0, w1, w2, 0.00001 , 0.01, f, d_w0, d_w1, d_w2)
    return (w0, w1, w2)

# Run
logisticRegression(DATASET)