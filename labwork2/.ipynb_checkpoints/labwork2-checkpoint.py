# Should use X and Y for list
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

# Can write derivative to w0 and w1 instead
def compute_gradients(x,y,w1,w0): 
    predictions = function(x,w1,w0)
    w1_gradient = sum((predictions[i] - y[i]) * x[i] for i in range(m)) / m
    w0_gradient = sum(predictions[i] - y[i] for i in range(m)) / m
    return w1_gradient, w0_gradient

def update_parameters(w1, w0, w1_gradient, w0_gradient, learning_rate):
    w1 = w1 - learning_rate * w1_gradient
    w0 = w0 - learning_rate * w0_gradient
    return w1, w0

def gradient_descent(x, y, w0, w1, num_iterations, learning_rate, threshold):    
    step = 0
    for i in range(num_iterations):
        w1_gradient, w0_gradient = compute_gradients(x, y, w1, w0)
        w1, w0 = update_parameters(w1, w0, w1_gradient, w0_gradient, learning_rate)
        
        loss = compute_loss(x, y, w1, w0)
        print(f"Iteration {i + 1}: Loss = {loss:<2.4f}, w1 = {w1:<2.4f}, w0 = {w0:<2.4f}, w1_grad = {w1_gradient:<2.4f}, w0_grad = {w0_gradient:<2.4f}\n" )
        
        if abs(w1_gradient) < threshold or abs(w0_gradient) < threshold:
            break
        step += 1
    
    return w1, w0, step

w1 = 0.0
w0 = 1.0
num_iterations = 100
learning_rate = 0.1
threshold = 1e-2
optimized_w1, optimized_w0, step = gradient_descent(x, y,w0 , w1, num_iterations, learning_rate, threshold)
print(f"Optimized w1: {optimized_w1:.4f}, w0: {optimized_w0:.4f}, step: {step}")

##################
# print(m)
# test = compute_loss(x,y,w1,w0)
# print(test)
# w1_grad_test,w0_grad_test = compute_gradients(x,y,w1,w0)
# print(w1_grad_test,w0_grad_test)
# w1_update_test,w0_update_test = update_parameters(w1, w0, w1_grad_test, w0_grad_test, learning_rate)
# print(w1_update_test,w0_update_test)
# loss = compute_loss(x,y,w1_update_test,w0_update_test)
# print(loss)