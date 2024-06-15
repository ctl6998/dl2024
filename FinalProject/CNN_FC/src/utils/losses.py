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
