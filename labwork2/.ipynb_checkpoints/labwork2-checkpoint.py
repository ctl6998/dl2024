x = []
y = []
with open('dataset_sam.csv', 'r') as f:
    results = []
    is_header = True
    for line in f:
        if is_header:
            is_header = False
            continue
        value = line.strip().split(',')
        x.append(float(value[0]))
        y.append(float(value[1]))
    
m = len(y)
    
def function(x,w1,w0):
    return [w1 * xi + w0 for xi in x]

def compute_loss(x,y,w1,w0):
    predictions = function(x,w1,w0)
    loss = sum((predictions[i] - y[i]) ** 2 for i in range(m)) / m
    return loss

def compute_gradients(x,y,w1,w0): 
    predictions = function(x,w1,w0)
    w1_gradient = sum((predictions[i] - y[i]) * x[i] for i in range(m)) / m
    w0_gradient = sum(predictions[i] - y[i] for i in range(m)) / m
    return w1_gradient, w0_gradient

def update_parameters(w1, w0, w1_gradient, w0_gradient, learning_rate):
    w1 = w1 - learning_rate * w1_gradient
    w0 = w0 - learning_rate * w0_gradient
    return w1, w0

def gradient_descent(x, y, num_iterations, learning_rate):
    w1 = 0.0
    w0 = 0.0
    
    for i in range(num_iterations):
        w1_gradient, w0_gradient = compute_gradients(x, y, w1, w0)
        w1, w0 = update_parameters(w1, w0, w1_gradient, w0_gradient, learning_rate)
        loss = compute_loss(x, y, w1, w0)
        print(f"Iteration {i + 1}: Loss = {loss:.4f}, w1 = {w1:.4f}, w0 = {w0:.4f}\n")
    
    return w1, w0

num_iterations = 1000
learning_rate = 10
optimized_w1, optimized_w0 = gradient_descent(x, y, num_iterations, learning_rate)
print(f"Optimized w1: {optimized_w1:.4f}, w0: {optimized_w0:.4f}")


