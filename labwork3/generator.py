import numpy as np
import pandas as pd

def generate_logistic_regression_data(num_samples, num_features, noise=0.1):
    # Generate random feature matrix (positive values with two decimal places)
    X = np.random.rand(num_samples, num_features).round(2)
    
    # Generate weights for features (positive values with two decimal places)
    true_weights = np.random.rand(num_features).round(2)
    
    # Generate labels using logistic function
    logits = np.dot(X, true_weights)
    prob = 1 / (1 + np.exp(-logits))
    labels = (prob > 0.5).astype(int)
    
    # Add noise to labels
    labels_with_noise = labels.copy()
    flip_indices = np.random.choice(num_samples, int(noise * num_samples), replace=False)
    labels_with_noise[flip_indices] = 1 - labels_with_noise[flip_indices]
    
    return X, labels_with_noise, true_weights

def save_to_csv(X, y, true_weights, filename):
    column_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
    column_names.append("Label")
    df = pd.DataFrame(np.column_stack([X, y]), columns=column_names)
    df.to_csv(filename, index=False)

# Parameters
num_samples = 10
num_features = 2
noise = 0.1
filename = "dataset.csv"

# Generate data
X, y, true_weights = generate_logistic_regression_data(num_samples, num_features, noise)

# Save to CSV
save_to_csv(X, y, true_weights, filename)
