import numpy as np
from typing import Literal, Callable

InitializationType = Literal["he", "xavier"]

def he_initialization(shape: tuple) -> np.ndarray:
    """
    He Initialization (also known as Kaiming Initialization).
    
    Designed specifically for layers with ReLU (Rectified Linear Unit) activations.
    Because ReLU zeroes out negative inputs (killing half the variance), 
    He initialization doubles the variance of the weights to maintain signal strength
    throughout the network.
    
    Formula: random_normal * sqrt(2 / input_dim)
    
    Args:
        shape (tuple): The shape of the weight matrix (output_dim, input_dim).
        
    Returns:
        np.ndarray: Initialized weight matrix.
    """
    D_i = shape[1] 
    return np.random.normal(size=shape) * np.sqrt(2 / D_i)


def xavier_initialization(shape: tuple) -> np.ndarray:
    """
    Xavier Initialization (also known as Glorot Initialization).
    
    Designed for layers with Sigmoid or Softmax activations.
    It attempts to keep the variance of the input and output gradients the same,
    keeping the signal within the linear region of the sigmoid function to 
    prevent vanishing gradients.
    
    Formula: random_normal * sqrt(1 / input_dim)
    
    Args:
        shape (tuple): The shape of the weight matrix (output_dim, input_dim).
        
    Returns:
        np.ndarray: Initialized weight matrix.
    """
    D_i = shape[1] 
    return np.random.normal(size=shape) * np.sqrt(1 / D_i)


def get_initialization(name: InitializationType) -> Callable[[tuple], np.ndarray]:
    """
    Factory function to retrieve the initialization strategy.
    
    Args:
        name (InitializationType): 'he' or 'xavier'.
        
    Returns:
        Callable: The corresponding initialization function.
        
    Raises:
        ValueError: If the initialization name is unknown.
    """
    if name == "xavier":
        return xavier_initialization
    if name == "he":
        return he_initialization
    raise ValueError(f"Unknown initialization function: {name}")