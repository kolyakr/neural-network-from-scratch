from src.activations import ActivationType, get_activation
from src.initialiaztion import get_initialization
from src.optimizers import OptimizerType, get_optimizer
from src.losses import get_loss, LossType
import numpy as np

class NeuralNetwork:
  """
    A fully-connected feedforward neural network implemented from scratch.

    Supports:
    - He/Xavier weight initialization
    - Arbitrary layer dimensions
    - ReLU / Sigmoid / Softmax / Linear activations
    - MSE, BCE, and CCE loss functions
    - GD / Momentum / Adam optimizers
    - Mini-batch training

    The network stores:
        - parameters (W, b)
        - chosen optimizer
        - chosen loss function
        - activation functions per layer
    
    Args:
        layer_dims (list[int]):
            A list defining the number of neurons per layer.
            Example: [4, 10, 5, 1] → 4 input features, two hidden layers, 1 output.

        activations (list[ActivationType]):
            A list of activation function names for each layer *except the input layer*.
            Length must be len(layer_dims) - 1.
            Example: ["relu", "relu", "sigmoid"].

        loss_type (LossType):
            One of {"mse", "bce", "cce"}.

        optimizer_type (OptimizerType):
            One of {"gd", "momentum", "adam"}.

        optimizer_params (dict):
            Keyword arguments passed to the optimizer.
            Example: {"beta": 0.9} for Momentum.

        seed (int):
            Random seed used for reproducible weight initialization.
    """
    
  def __init__(self, 
               layer_dims: list[int], 
               activations: list[ActivationType], 
               loss_type=LossType, 
               optimizer_type : OptimizerType = "gd",
               optimizer_params: dict = None,
               seed : int = 42):
    self._validate_inputs(layer_dims, activations, loss_type, optimizer_type)
    
    self.layer_dims = layer_dims
    self.activations = activations
    self.loss_type = loss_type
    self.optimizer_type = optimizer_type
    self.optimizer_params = optimizer_params if optimizer_params else {}
    self.seed = seed
    self.params = {}
    
    self._initialize_parameters()
    self.optimizer = get_optimizer(self.optimizer_type, **self.optimizer_params)
    
    
  def forward_pass(self, X: np.ndarray):
    """
    Perform the forward propagation step.

    Computes:
        Z[l] = W[l] · A[l-1] + b[l]
        A[l] = activation(Z[l])

    Args:
        X (np.ndarray):
            Input matrix of shape (input_dim, m),
            where m is the number of samples.

    Returns:
        tuple:
            A (np.ndarray):
                Final output of the network (y_hat).
            caches (list):
                A list of tuples (A_prev, Z) for each layer, 
                used during backpropagation.

    Raises:
        ValueError:
            If the input dimensionality does not match the network definition.
    """
    
    # Validation
    if self.layer_dims[0] != X.shape[0]:
        raise ValueError(
            f"Input dimension mismatch. Expected {self.layer_dims[0]} features, "
            f"but got {X.shape[0]}."
        )

    L = len(self.layer_dims)
    A = X
    caches = [] 
    
    for i in range(1, L):
        act_name = self.activations[i - 1]
        act_obj = get_activation(act_name)
        
        if isinstance(act_obj, tuple):
            act_fnc = act_obj[0]
        else:
            act_fnc = act_obj

        W = self.params[f"W{i}"]
        b = self.params[f"b{i}"]
        A_prev = A
        
        Z = np.dot(W, A_prev) + b 
        A = act_fnc(Z)
        
        caches.append((A_prev, Z))

    return A, caches

  def backward_pass(self, y_true: np.ndarray, y_hat: np.ndarray, caches: list) -> dict:
    """
    Perform backpropagation to compute gradients of all parameters.

    Supports:
        - Automatic derivative of the loss function
        - Activation derivatives where applicable
        - BCE/CCE stability trick (using dZ = y_hat - y_true)

    Args:
        y_true (np.ndarray):
            Ground truth labels, shape (output_dim, m).

        y_hat (np.ndarray):
            Predicted outputs from forward pass.

        caches (list):
            Cached (A_prev, Z) values from forward_pass.

    Returns:
        grads (dict):
            Dictionary containing gradients:
                dL_dW1, dL_db1, ..., dL_dWL, dL_dbL

        loss (float):
            Scalar loss computed using the chosen loss function.

    Raises:
        ValueError:
            If y_true and y_hat have mismatched shapes.
            If an activation function has no defined derivative.
    """
    
    if len(y_true) != len(y_hat):
        raise ValueError(f"y_true and y_hat must have the same shapes."
                          f"Got: y_true{y_true.shape}, y_hat{y_hat.shape}")
    
    L = len(self.layer_dims)
    m = y_true.shape[1]
    
    grads = {}
    loss = get_loss(self.loss_type)[0](y_hat, y_true)
    
    #compute for last dA and dZ
    A, Z = caches[L - 2]
    
    if self.loss_type in ["bce", "cce"]:  
        dZ = y_hat - y_true
    else:
        loss_derivative = get_loss(self.loss_type)[1]
        dA = loss_derivative(y_hat, y_true)
        
        act_name = self.activations[-1]
        act_obj = get_activation(act_name)
        
        if(isinstance(act_obj, tuple)):
            act_derivative = act_obj[1]
            dZ = dA * act_derivative(Z)
        else:
            dZ = dA
            
    grads[f"dL_dW{L - 1}"] = (1/m) * np.dot(dZ, A.T)
    grads[f"dL_db{L - 1}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    
    dA = np.dot(self.params[f"W{L - 1}"].T, dZ)
    
    for l in range( L - 2, 0, -1):
        A, Z = caches[l - 1]
        
        act_obj = get_activation(self.activations[l - 1])
    
        if(isinstance(act_obj, tuple)):
          act_derivative = act_obj[1]
          dZ = act_derivative(Z) * dA
        else:
          raise ValueError(f"Activation {act_name} has no derivative defined.") 
        
        grads[f"dL_dW{l}"] = (1/m) * np.dot(dZ, A.T)
        grads[f"dL_db{l}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        if(l > 1):
          dA = np.dot(self.params[f"W{l}"].T, dZ)
    
    return grads, loss
  
  def train(self,
          X_train: np.ndarray,
          y_train: np.ndarray,
          epochs: int = 1000,
          learning_rate: float = 0.01,
          batch_size: int = None,):
    """
    Train the neural network using mini-batch gradient descent.

    Steps:
        1) Shuffle the dataset each epoch
        2) Split data into batches
        3) Forward pass
        4) Backward pass
        5) Parameter update (via the chosen optimizer)
        6) Track loss over time

    Args:
        X_train (np.ndarray):
            Training input data of shape (input_dim, m).

        y_train (np.ndarray):
            Training labels of shape (output_dim, m).

        epochs (int):
            Number of passes over the full training set.

        learning_rate (float):
            Step size for the optimizer.

        batch_size (int or None):
            If None → full batch gradient descent.
            Otherwise → mini-batch gradient descent.

    Returns:
        list[float]:
            Loss values collected after each batch.

    Raises:
        ValueError:
            If input-output sample counts mismatch.
            If batch_size is larger than the dataset.
    """
  
    if(X_train.shape[1] != y_train.shape[1]):
        raise ValueError(f"X_train and y_train must have the same size of observations."
                        f"Got:"
                        f"X_train: {X_train.shape[1]}"
                        f"y_train: {y_train.shape[1]}")
        
    n = X_train.shape[1]
        
    if(batch_size != None and batch_size > n):
        raise ValueError(f"batch_size must not be bigger than the number of observations in dataset"
                        f"Got:"
                        f"X_train: {X_train.shape[1]}"
                        f"batch_size: {batch_size}")
        
    losses = []
    idx = np.arange(0, n)
    
    for i in range(0, epochs):
        
      permutation = np.random.permutation(idx)
      X_shuffled = X_train[:, permutation]
      y_shuffled = y_train[:, permutation]
      
      for j in range(0, n, batch_size):
      
        y_hat, caches = self.forward_pass(X_shuffled[:, j: j + batch_size])
        grads, batch_loss = self.backward_pass(y_shuffled[:, j: j + batch_size], y_hat, caches)
        self.optimizer.update(
            self.params,
            grads,
            learning_rate
        )
        
        losses.append(batch_loss)
    
    return losses
              
  def _initialize_parameters(self):
    """
    Initialize all network weights and biases.

    Weight initialization strategies:
        - ReLU layers → He initialization
        - Sigmoid/Softmax/Linear layers → Xavier initialization

    Biases are initialized to zeros.

    Uses the provided random seed for reproducibility.
    """
    
    np.random.seed(self.seed)
    
    L = len(self.layer_dims)
    
    for i in range(1, L):
      D_o =  self.layer_dims[i]
      D_i = self.layer_dims[i - 1]
      act_fnc = self.activations[i - 1]
      
      if act_fnc == "relu":
        he_initialization = get_initialization("he")
        self.params[f"W{i}"] = he_initialization((D_o, D_i))
      if act_fnc == "sigmoid" or act_fnc == "softmax" or act_fnc == "linear":
        xavier_initialization = get_initialization("xavier")
        self.params[f"W{i}"] = xavier_initialization((D_o, D_i))
      
      self.params[f"b{i}"] = np.zeros((D_o, 1))
          
  def _validate_inputs(self, layer_dims, activations, loss_type, optimizer_type):
    """
    Private helper to validate all inputs before initialization.
    """
    if not isinstance(layer_dims, list):
        raise TypeError(f"layer_dims must be a list, got {type(layer_dims)}")
    
    if not all(isinstance(x, int) for x in layer_dims):
        raise TypeError("All elements in layer_dims must be integers!")

    if not isinstance(activations, list):
          raise TypeError(f"activations must be a list, got {type(activations)}")

    if len(layer_dims) < 2:
        raise ValueError("The length of layers must be at least 2 (Input -> Output)") 
    
    if min(layer_dims) < 1:
          raise ValueError("The number of neurons in every layer must be at least 1")
    
    if len(layer_dims) != len(activations) + 1:
          raise ValueError(
              f"Structure Error: You provided {len(layer_dims)} layers but {len(activations)} activations. "
              f"Expected {len(layer_dims) - 1} activations."
          )

    valid_activations = {"relu", "sigmoid", "softmax", "linear"}
    for act in activations:
      if act not in valid_activations:
        raise ValueError(f"Invalid activation '{act}'. Supported: {valid_activations}")

    valid_losses = {"mse", "bce", "cce"}
    if loss_type not in valid_losses:
      raise ValueError(f"Invalid loss_type '{loss_type}'. Supported: {valid_losses}")