DATASET = []
with open('dataset_sam.csv', 'r') as f:
    next(f)  # Skip the header
    for line in f:
        x, y = map(float, line.strip().split(','))
        DATASET.append((x, y))

# gradient descent
def gradientDescend2d(param0, param1, L, t, f, d_w0, d_w1):
    w0 = param0 
    w1 = param1 
    while True:
        oldF = f(w0, w1)
        w0 = w0 - L * d_w0(w0, w1)
        w1 = w1 - L * d_w1(w0, w1)
        newF = f(w0, w1)
        diff = abs(newF - oldF)
        print(f"{round(w0, 2)}\t\t{round(w1, 2)}\t\t{round(newF, 2)}\t\t{round(oldF, 2)}\t\t{round(diff, 2)}")
        if diff < t:
            print("Finished")
            return (w0, w1)
        
# diff functions
diff = lambda w0, w1, xi, yi: 1/2 * (w1 * xi + w0 - yi) ** 2
diff_w0 = lambda w0, w1, xi, yi: w1 * xi + w0 - yi
diff_w1 = lambda w0, w1, xi, yi: xi * (w1 * xi + w0 - yi)

# loss functions 
loss = lambda dataset, w0, w1: sum( [diff(w0, w1, data[0], data[1]) for data in dataset] ) / len(DATASET)
loss_w0 = lambda dataset, w0, w1: sum( [diff_w0(w0, w1, data[0], data[1]) for data in dataset] ) / len(DATASET)
loss_w1 = lambda dataset, w0, w1: sum( [diff_w1(w0, w1, data[0], data[1]) for data in dataset] ) / len(DATASET)

def linearRegression(dataset):
    w0, w1 = 0, 1
    d = lambda w0, w1: loss(dataset, w0, w1)
    d_w0 = lambda w0, w1: loss_w0(dataset, w0, w1)
    d_w1 = lambda w0, w1: loss_w1(dataset, w0, w1)
    (w0, w1) = gradientDescend2d(w0, w1, 0.1, 0.001, d, d_w0, d_w1)
    return (w0, w1)
               
# run
linearRegression(DATASET)
        