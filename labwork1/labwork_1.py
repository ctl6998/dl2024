import time

def function(x):
    return x ** 2

def derivative(x):
    return 2 * x

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
        
        print(f"{step} | {exec_time:<2.10f} | {x:<2.10f} | {f_x:<2.16f}")
        
        x = x - learning_rate * f_derivative_x
        if abs(f_derivative_x) < threshold:
            break
            
        step += 1
    
    print("Finish. Steps required: ",step)

starting_point = 10.0 
learning_rate = 0.1  
threshold = 1e-6 
max_step = 100 

gradient_descent(starting_point, learning_rate, threshold, max_step)
