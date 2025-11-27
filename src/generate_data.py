from sklearn.datasets import make_moons
import numpy as np

def get_classification_data(n_samples=1000, noise=0.1, seed=42):
    """
    Generates a binary classification dataset (Moons).
    Returns shapes: X=(2, m), y=(1, m)
    """
    np.random.seed(seed)
    
    X_raw, y_raw = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    
    # --- TRANSFORMATION FOR YOUR NN CLASS ---
    # 1. Transpose X to get (features, samples)
    X = X_raw.T 
    
    # 2. Reshape y to (1, samples)
    y = y_raw.reshape(1, -1)
    
    return X, y

def get_regression_data(n_samples=1000, seed=42):
    """
    Generates a non-linear regression dataset (Noisy Sine Wave).
    Returns shapes: X=(1, m), y=(1, m)
    """
    np.random.seed(seed)
    
    # Generate X values between -5 and 5
    # Shape: (1, m)
    X = np.random.uniform(-5, 5, (1, n_samples))
    
    # Generate y = sin(x) + Gaussian Noise
    # Shape: (1, m)
    noise = np.random.normal(0, 0.1, (1, n_samples))
    y = np.sin(X) + noise
    
    # Normalize X to range [0, 1] or [-1, 1] usually helps NN convergence,
    # but strictly for generation we leave it raw here.
    # Note: Neural Nets struggle with unscaled data. 
    # It is recommended to scale X before training.
    
    return X, y