import time

#### f(x) = x^2
# def function(x):
#     return x ** 2

# def derivative(x):
#     return 2 * x

#### f(x) = x^4
def function(x):
    return x**4

def derivative(x):
    return 4 * x**3

def gradient_descent(starting_point, learning_rate, threshold, max_step):
    x = starting_point
    start_time = time.time()
    step = 0
    
    print("Step    |    Time(s)    |       x       |      f(x)     ")
    print("--------|---------------|---------------|---------------")
    
    while step < max_step:
        f_x = function(x)
        f_derivative_x = derivative(x)
        
        ite_start_time = time.time()
        exec_time = ite_start_time - start_time
        
        print(f"{step} | {exec_time:<2.4f} | {x:<2.4f} | {f_x:<2.4f} | {f_derivative_x:<2.4f}")
        
        x = x - learning_rate * f_derivative_x
        if abs(f_derivative_x) < threshold:
            break
            
        step += 1
    
    total_time = time.time() - start_time
    print("Finish. Steps required: ",step)
    print(f"Total execution time: {total_time:.10f} seconds")

starting_point = 10.0 
learning_rate = 0.004 
threshold = 1e-2 
max_step = 10000

gradient_descent(starting_point, learning_rate, threshold, max_step)
