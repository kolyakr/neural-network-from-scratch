import numpy as np
from src.activations import get_activation
from typing import Literal, Callable

LossType = Literal["mse", "bce", "cce"]
LossFunc = Callable[[np.ndarray, np.ndarray], float]
Derivative = Callable[[np.ndarray, np.ndarray], np.ndarray]

sigmoid_func = get_activation("sigmoid")[0] 
softmax_func = get_activation("softmax")

def mse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).
    
    Used for Regression.
    
    Args:
        y_hat (np.ndarray): The final active predictions.
        y_true (np.ndarray): The actual target values.
        
    Returns:
        float: The sum of squared errors.
    """
    return np.mean((y_hat - y_true)**2)

def mse_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of MSE.
    
    Args:
        y_hat (np.ndarray): Predictions.
        y_true (np.ndarray): Targets.
        
    Returns:
        np.ndarray: Gradient vector.
    """
    return 2 * (y_hat - y_true) / y_true.size

def binary_cross_entropy(z: np.ndarray, y_true: np.ndarray) -> float:
    """
    Binary Cross Entropy (BCE).
    
    Used for Binary Classification (0 or 1).
    EXPECTS LOGITS (z) as input, not probabilities. This function applies 
    Sigmoid internally for numerical stability.
    
    Args:
        z (np.ndarray): Raw logits (pre-activation).
        y_true (np.ndarray): Targets (0 or 1).
        
    Returns:
        float: The loss value.
    """
    p = sigmoid_func(z)
    
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
    
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def binary_cross_entropy_derivative(z: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of BCE with respect to Logits (z).
    
    Formula simplifiction: (Sigmoid(z) - y)
    
    Args:
        z (np.ndarray): Raw logits.
        y_true (np.ndarray): Targets.
        
    Returns:
        np.ndarray: Gradient vector.
    """
    return sigmoid_func(z) - y_true

def categorical_cross_entropy(z: np.ndarray, y_true: np.ndarray) -> float:
    """
    Categorical Cross Entropy (CCE).
    
    Used for Multi-class Classification.
    EXPECTS LOGITS (z) as input. Applies Softmax internally.
    
    Args:
        z (np.ndarray): Raw logits (N samples, C classes).
        y_true (np.ndarray): One-hot encoded targets (N, C).
        
    Returns:
        float: The loss value.
    """
    p = softmax_func(z)
    
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1.0)
    
    return -np.mean(np.sum((y_true * np.log(p)), axis=1))

def categorical_cross_entropy_derivative(z: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of CCE with respect to Logits (z).
    
    Formula simplification: (Softmax(z) - y)
    
    Args:
        z (np.ndarray): Raw logits.
        y_true (np.ndarray): One-hot encoded targets.
        
    Returns:
        np.ndarray: Gradient vector.
    """
    return softmax_func(z) - y_true

def get_loss(name: LossType) -> tuple[LossFunc, Derivative]:
    """
    Factory function to retrieve loss and derivative functions.
    
    Args:
        name (LossType): 'mse', 'bce', or 'cce'.
        
    Returns:
        tuple: (Loss Function, Derivative Function)
    """
    if name == "mse":
        return (mse, mse_derivative)
    if name == "bce":
        return (binary_cross_entropy, binary_cross_entropy_derivative)
    if name == "cce":
        return (categorical_cross_entropy, categorical_cross_entropy_derivative)
    raise ValueError(f"Unknown loss: {name}")