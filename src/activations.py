import numpy as np
from typing import Callable, Literal, Union

ActivationType = Literal["relu", "sigmoid", "softmax", "linear"]
ActFunc = Callable[[np.ndarray], np.ndarray]

def relu(z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Introduces non-linearity by clipping negative values to zero, allowing the network to learn
    complex patterns.
    
    Args:
        z (np.ndarray): Pre-activation input (any shape, e.g., (N, units) for batches).
    
    Returns:
        np.ndarray: Activated output, same shape as z.
    """
    return z.clip(0.0,)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of the ReLU function.
    
    Used in backpropagation to compute gradients.
    Returns 1 for positive z, 0 otherwise (undefined at 0, but approximated as 0).
    
    Args:
        z (np.ndarray): Pre-activation input (same shape as in forward pass).
    
    Returns:
        np.ndarray: Derivative values, same shape as z.
    """
    return (z > 0).astype(float)

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Maps inputs to (0, 1), useful for binary classification or probabilistic outputs.
    Can suffer from vanishing gradients in deep networks.
    
    Args:
        z (np.ndarray): Pre-activation input.
    
    Returns:
        np.ndarray: Activated output in (0, 1).
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of the sigmoid function.
    
    Computed efficiently using the sigmoid output itself for backprop.
    
    Args:
        z (np.ndarray): Pre-activation input.
    
    Returns:
        np.ndarray: Derivative values.
    """
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax activation function.
    
    Converts logits to probabilities summing to 1, for multiclass classification outputs
    (ref: "Loss functions.pdf" for maximum likelihood with categorical distributions).
    Assumes z is (N, C) for N samples and C classes; for 1D, it treats as single sample but may not sum correctly—consider reshaping in caller.
    
    Note: No separate derivative; use (y_hat - y_true) in backprop for cross-entropy.
    
    Args:
        z (np.ndarray): Logits (pre-activation), typically (N, C).
    
    Returns:
        np.ndarray: Probabilities, same shape as z, rows sum to 1.
    """
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0)

def linear(z: np.ndarray) -> np.ndarray:
    """
    Linear (Identity) activation function.
    Used for the output layer in Regression tasks.
    Returns input unchanged.
    """
    return z

def linear_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of Linear function.
    The slope of y = x is always 1.
    """
    return np.ones_like(z)

def get_activation(name: ActivationType) -> Union[tuple[ActFunc, ActFunc], ActFunc]:
    """
    Factory function to retrieve activation and its derivative by name.
    
    For hidden layers (relu, sigmoid), returns (func, deriv). For output (softmax), returns func only.
    Used in NeuralNetwork to configure layers dynamically.
    
    Args:
        name (ActivationType): Name of the activation ("relu", "sigmoid", "softmax", "linear").
    
    Returns:
        Union[tuple[ActFunc, ActFunc], ActFunc]: Function(s) for forward/backward passes.
    
    Raises:
        ValueError: If name is invalid.
    """
    if name == "relu":
        return (relu, relu_derivative)
    if name == "sigmoid":
        return (sigmoid, sigmoid_derivative)
    if name == "softmax":
        return (softmax)
    if name == "linear":
        return (linear, linear_derivative)
    raise ValueError(f"Unknown activation: {name}")