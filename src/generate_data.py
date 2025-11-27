from sklearn.datasets import make_moons
import numpy as np

def get_classification_data(n_samples=1000, noise=0.1, seed=42):
    """
    Generates a binary classification dataset (Moons).
    Returns shapes: X=(2, m), y=(1, m)
    """
    np.random.seed(seed)
    
    X_raw, y_raw = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    
    X = X_raw.T 
    
    y = y_raw.reshape(1, -1)
    
    return X, y

def get_regression_data(n_samples=1000, seed=42):
    """
    Generates a non-linear regression dataset (Noisy Sine Wave).
    Returns shapes: X=(1, m), y=(1, m)
    """
    np.random.seed(seed)
    
    X = np.random.uniform(-5, 5, (1, n_samples))
    
    noise = np.random.normal(0, 0.1, (1, n_samples))
    y = np.sin(X) + noise
    
    return X, y