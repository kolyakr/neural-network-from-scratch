import numpy as np
from typing import Literal, Callable

LossType = Literal["mse", "bce", "cce"]
LossFunc = Callable[[np.ndarray, np.ndarray], float]
Derivative = Callable[[np.ndarray, np.ndarray], np.ndarray]

def mse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).
    
    Used for Regression.
    
    Formula:
        $$ L = \\frac{1}{N} \\sum (\\hat{y} - y)^2 $$

    Args:
        y_hat (np.ndarray): The final active predictions.
        y_true (np.ndarray): The actual target values.
        
    Returns:
        float: The sum of squared errors.
    """
    return np.mean(np.square(y_hat - y_true))

def mse_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of MSE with respect to y_hat.
    
    Formula:
        $$ \\frac{\\partial L}{\\partial \\hat{y}} = \\frac{2}{N} (\\hat{y} - y) $$
    
    Args:
        y_hat (np.ndarray): Predictions.
        y_true (np.ndarray): Targets.
        
    Returns:
        np.ndarray: Gradient vector.
    """
    return 2 * (y_hat - y_true) 

def binary_cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Binary Cross Entropy (BCE).
    
    Expects probabilities (y_hat), not logits.
    
    Formula:
        $$ L = - \\frac{1}{N} \\sum [y \\cdot \\log(\\hat{y}) + (1 - y) \\cdot \\log(1 - \\hat{y})] $$
    
    Args:
        y_hat (np.ndarray): Activated predictions (after Sigmoid).
        y_true (np.ndarray): Targets (0 or 1).
        
    Returns:
        float: The loss value.
    """
    epsilon = 1e-15
    # Clip to prevent log(0)
    p = np.clip(y_hat, epsilon, 1 - epsilon)
    
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def binary_cross_entropy_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of BCE with respect to y_hat.
    
    Formula:
        $$ \\frac{\\partial L}{\\partial \\hat{y}} = - \\frac{y}{\\hat{y}} + \\frac{1 - y}{1 - \\hat{y}} $$
    
    Args:
        y_hat (np.ndarray): Predictions (probabilities).
        y_true (np.ndarray): Targets.
        
    Returns:
        np.ndarray: Gradient vector (dA).
    """
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return (-(y_true / y_hat) + ((1 - y_true) / (1 - y_hat)))

def categorical_cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Categorical Cross Entropy (CCE).
    
    Expects probabilities (y_hat).
    
    Formula:
        $$ L = - \\frac{1}{N} \\sum \\sum y \\cdot \\log(\\hat{y}) $$
    
    Args:
        y_hat (np.ndarray): Activated predictions (after Softmax).
        y_true (np.ndarray): One-hot encoded targets.
        
    Returns:
        float: The loss value.
    """
    epsilon = 1e-15
    # Clip to prevent log(0)
    p = np.clip(y_hat, epsilon, 1.0)
    
    return -np.mean(np.sum((y_true * np.log(p)), axis=1))

def categorical_cross_entropy_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of CCE with respect to y_hat.
    
    Formula:
        $$ \\frac{\\partial L}{\\partial \\hat{y}} = - \\frac{y}{\\hat{y}} $$
    
    Args:
        y_hat (np.ndarray): Predictions (probabilities).
        y_true (np.ndarray): One-hot encoded targets.
        
    Returns:
        np.ndarray: Gradient vector (dA).
    """
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1.0)
    return - (y_true / y_hat)

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