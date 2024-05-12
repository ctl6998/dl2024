from math import log, exp

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
        
def gradientDescend(n