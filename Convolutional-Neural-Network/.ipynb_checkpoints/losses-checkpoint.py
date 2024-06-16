import math

def mse(y_true, y_pred):
    total_loss = 0.0
    count = 0
    for t, p in zip(y_true, y_pred):
        for true, pred in zip(t, p):
            total_loss += (true[0] - pred[0]) ** 2 if isinstance(true, list) and isinstance(pred, list) else (true - pred) ** 2
            count += 1
    return total_loss / count

def mse_derivative(y_true, y_pred):
    derivatives = []
    count = len(y_true) * len(y_true[0])
    for t, p in zip(y_true, y_pred):
        layer_derivative = []
        for true, pred in zip(t, p):
            if isinstance(true, list) and isinstance(pred, list):
                layer_derivative.append([(2 * (pred[0] - true[0])) / count])
            else:
                layer_derivative.append((2 * (pred - true)) / count)
        derivatives.append(layer_derivative)
    return derivatives

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-12  # to avoid log(0)
    total_loss = 0.0
    count = 0
    for t, p in zip(y_true, y_pred):
        for true, pred in zip(t, p):
            pred_value = pred[0] if isinstance(pred, list) else pred
            pred_value = max(min(pred_value, 1. - epsilon), epsilon)
            true_value = true[0] if isinstance(true, list) else true
            total_loss += -true_value * math.log(pred_value) - (1 - true_value) * math.log(1 - pred_value)
            count += 1
            
    return total_loss / count

def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-12  # to avoid division by 0
    derivatives = []
    count = len(y_true) * len(y_true[0])
    for t, p in zip(y_true, y_pred):
        layer_derivative = []
        for true, pred in zip(t, p):
            pred_value = pred[0] if isinstance(pred, list) else pred
            pred_value = max(min(pred_value, 1. - epsilon), epsilon)
            true_value = true[0] if isinstance(true, list) else true
            derivative = ((pred_value - true_value) / (pred_value * (1. - pred_value))) / count
            layer_derivative.append(derivative)
        derivatives.append(layer_derivative)
    return derivatives

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-12  
    total_loss = 0.0
    count = 0
    for t, p in zip(y_true, y_pred):
        for true, pred in zip(t, p):
            pred_value = pred[0] if isinstance(pred, list) else pred
            pred_value = max(min(pred_value, 1. - epsilon), epsilon)
            true_value = true[0] if isinstance(true, list) else true
            total_loss += -true_value * math.log(pred_value)
            count += 1
    return total_loss / count

def categorical_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-12  
    derivatives = []
    count = len(y_true) * len(y_true[0])
    for t, p in zip(y_true, y_pred):
        layer_derivative = []
        for true, pred in zip(t, p):
            pred_value = pred[0] if isinstance(pred, list) else pred
            pred_value = max(min(pred_value, 1. - epsilon), epsilon)
            true_value = true[0] if isinstance(true, list) else true
            derivative = -(true_value / pred_value) / count
            layer_derivative.append(derivative)
        derivatives.append(layer_derivative)
    return derivatives


#Another version of binary cross entropy written for Softmax, as the output of Softmax is in list form
def binary_cross_entropy_softmax(y_actual, y_predicted):
    avoidance = 1e-12 
    total_loss = 0.0
    total_count = 0
    
    if not isinstance(y_predicted, (list, tuple)):
        y_predicted = [y_predicted]
    
    for actual, predicted in zip(y_actual, y_predicted):
        actual_value = actual[0] if isinstance(actual, (list, tuple)) else actual
        predicted_value = predicted
        predicted_value = max(min(predicted_value, 1. - avoidance), avoidance)
        loss = -actual_value * math.log(predicted_value) - (1 - actual_value) * math.log(1 - predicted_value)
        total_loss += loss
        total_count += 1
    average_loss = total_loss / total_count if total_count > 0 else 0.0
    return average_loss

def binary_cross_entropy_derivative_softmax(y_actual, y_predicted):
    avoidance = 1e-12  
    derivatives = []
    if not isinstance(y_actual, (list, tuple)):
        y_actual = [y_actual]
    if not isinstance(y_predicted, (list, tuple)):
        y_predicted = [y_predicted]
    
    total_count = len(y_actual)
    
    for actual_value, predicted_value in zip(y_actual, y_predicted):
        if isinstance(predicted_value, (list, tuple)):
            pred_val = predicted_value[0]
        else:
            pred_val = predicted_value

        pred_val = max(min(pred_val, 1. - avoidance), avoidance)

        if isinstance(actual_value, (list, tuple)):
            actual_val = actual_value[0]
        else:
            actual_val = actual_value

        derivative = ((pred_val - actual_val) / (pred_val * (1. - pred_val))) / total_count
        derivatives.append(derivative)
        if len(derivatives) == 1:
            derivatives = derivatives[0]
    
    return derivatives
