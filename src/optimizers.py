import numpy as np
from typing import Dict, Literal, Callable

OptimizerType = Literal["gd", "momentum", "adam"]


class Optimizer:
    """
    Base class for all optimization algorithms.
    
    Each optimizer must implement the `update` method, which applies a parameter
    update step based on gradients computed during backpropagation.
    """
    def __init__(self):
        pass

    def update(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        learning_rate: float
    ):
        """
        Apply a parameter update step.
        
        Args:
            params (Dict[str, np.ndarray]): Model parameters (e.g., weights, biases).
            grads (Dict[str, np.ndarray]): Corresponding gradients (keys follow 'dL_param').
            learning_rate (float): Step size for the update.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        return NotImplementedError("Subclasses must implement 'update'")


class GradientDescent(Optimizer):
    """
    Standard Gradient Descent (GD).
    
    Updates each parameter by moving it in the direction opposite to its gradient.
    This is the simplest optimization method.
    
    Update rule:
        $$ \\theta \\leftarrow \\theta - \\alpha \\cdot \\nabla L(\\theta) $$
    """
    def update(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        learning_rate: float
    ):
        for key in params:
            grad_key = f"dL_d{key}"
            if grad_key in grads:
                params[key] -= learning_rate * grads[grad_key]


class Momentum(Optimizer):
    """
    Gradient Descent with Momentum.
    
    Momentum accelerates learning by accumulating an exponentially decaying
    moving average of past gradients. This helps reduce oscillations and speeds up
    convergence in valleys or ravines.
    
    Update rules:
        $$ v \\leftarrow \\beta v + (1 - \\beta) \\cdot g $$
        $$ \\theta \\leftarrow \\theta - \\alpha \\cdot v $$
    
    Args:
        beta (float): Momentum decay factor ($0 \\le \\beta \\le 1$).
    """
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.velocities: Dict[str, float] = {}

        if self.beta > 1 or self.beta < 0:
            raise ValueError("Beta's value must be in range [0, 1]")

    def update(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        learning_rate: float
    ):
        for key in params:
            grad_key = f"dL_d{key}"
            old_velocity = self.velocities.get(key, 0)

            if grad_key in grads:
                new_velocity = self.beta * old_velocity + (1 - self.beta) * grads[grad_key]
                params[key] -= learning_rate * new_velocity
                self.velocities[key] = new_velocity


class Adam(Optimizer):
    """
    Adam Optimizer (Adaptive Moment Estimation).
    
    Combines the benefits of Momentum (first moment) and RMSProp (second moment).
    Maintains exponentially weighted averages of both gradients and squared gradients.
    Uses bias correction to stabilize early updates.
    
    Update rules:
        $$ m \\leftarrow \\beta_1 \\cdot m + (1 - \\beta_1) \\cdot g $$
        $$ v \\leftarrow \\beta_2 \\cdot v + (1 - \\beta_2) \\cdot g^2 $$
        $$ \\hat{m} \\leftarrow \\frac{m}{1 - \\beta_1^t} $$
        $$ \\hat{v} \\leftarrow \\frac{v}{1 - \\beta_2^t} $$
        $$ \\theta \\leftarrow \\theta - \\alpha \\cdot \\frac{\\hat{m}}{\\sqrt{\\hat{v}} + \\epsilon} $$
    
    Args:
        beta (float): First-moment decay term $\\beta_1$.
        hamma (float): Second-moment decay term $\\beta_2$.
        epsilon (float): Small constant to avoid division by zero.
    """
    def __init__(self, beta: float = 0.9, hamma: float = 0.999, epsilon: float = 1e-8):
        self.beta = beta
        self.hamma = hamma
        self.t = 0
        self.epsilon = epsilon

        if not (0 <= self.beta <= 1):
            raise ValueError("Parameter beta must be in range [0, 1]")
        if not (0 <= self.hamma <= 1):
            raise ValueError("Parameter hamma must be in range [0, 1]")

        self.m = {}
        self.v = {}

    def update(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        learning_rate: float
    ):
        self.t += 1

        for key in params:
            grad_key = f"dL_d{key}"
            old_momentum = self.m.get(key, 0)
            old_velocity = self.v.get(key, 0)

            if grad_key in grads:
                g = grads[grad_key]

                # First and second moments
                new_momentum = self.beta * old_momentum + (1 - self.beta) * g
                new_velocity = self.hamma * old_velocity + (1 - self.hamma) * np.power(g, 2)

                # Bias correction
                m_hat = new_momentum / (1 - np.power(self.beta, self.t))
                v_hat = new_velocity / (1 - np.power(self.hamma, self.t))

                # Parameter update
                params[key] -= learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))

                self.m[key] = new_momentum
                self.v[key] = new_velocity


def get_optimizer(name: OptimizerType, **kwargs) -> Optimizer:
    """
    Factory function that returns an optimizer instance.
    
    Args:
        name (OptimizerType): One of ["gd", "momentum", "adam"].
        **kwargs: Additional parameters passed to optimizer constructors.
    
    Returns:
        Optimizer: An initialized optimizer object.
    
    Raises:
        ValueError: If the optimizer name is unknown.
    """
    if name == "gd":
        return GradientDescent()
    if name == "momentum":
        return Momentum(**kwargs)
    if name == "adam":
        return Adam(**kwargs)

    raise ValueError(f"Unknown optimizer: {name}")