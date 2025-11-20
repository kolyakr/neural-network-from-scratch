## Project Structure Template

The structure assumes a Python environment with NumPy (and optionally Matplotlib for visualizations, scikit-learn for data generation). No external ML frameworks like TensorFlow or PyTorch are used – everything is from scratch. You'll write the code yourself, but I've outlined placeholders, key functions/classes, and what to implement in each file based on your plan.

### Directory Layout

```
numpy-deep-learning-project/
├── src/                  # Core Python modules for the neural network implementation
│   ├── __init__.py       # Makes src a package (optional, for imports)
│   ├── neural_network.py # Main NeuralNetwork class and training logic
│   ├── activations.py    # Activation functions and derivatives
│   ├── losses.py         # Loss functions and derivatives
│   └── optimizers.py     # Optimizer classes/implementations
├── notebooks/            # Jupyter notebooks for experiments and demos
│   └── train.ipynb       # Main notebook for data prep, training, evaluation, and visualizations
├── data/                 # Optional: Store generated datasets (e.g., CSV files from make_moons)
│   └── (e.g., moons_train.csv, moons_test.csv)
├── docs/                 # Documentation files (for proper/parallel documentation)
│   └── architecture.md   # High-level diagrams, explanations (e.g., using Markdown or Draw.io)
├── README.md             # Project overview, setup instructions, usage
├── requirements.txt      # Dependencies (e.g., numpy, matplotlib, scikit-learn)
└── .gitignore            # Ignore pycache, notebooks checkpoints, etc.
```

### What Each Component Contains

I'll describe each file/directory in detail, including key elements to implement. Focus on proper documentation:

- Use **docstrings** (PEP 257 style) for classes/functions to explain purpose, parameters, returns.
- Add **inline comments** for complex logic (e.g., math derivations from the provided PDFs like "Gradients and initialization.pdf").
- Document **parallelly/properly** by updating docs/architecture.md as you code – e.g., add explanations referencing the PDFs (e.g., "Backpropagation inspired by 'Gradients and initialization.pdf' Page 1-2").
- Use type hints (e.g., `def func(x: np.ndarray) -> np.ndarray`) for clarity.

#### 1. `src/neural_network.py`

- **Purpose**: Core class for the configurable Deep Neural Network (DNN). Handles initialization, forward/backward passes, parameter updates, and training loop. Supports regression/classification via configurable activations/losses/optimizers.
- **Key Contents**:
  - `class NeuralNetwork`:
    - `__init__(self, layer_dims: list[int], activations: list[str], loss_type: str, optimizer_type: str, seed=42)`: Initialize parameters (He/Xavier init from "Gradients and initialization.pdf"), store activations/loss/optimizer.
    - `_initialize_params(self)`: Random initialization for weights/biases (e.g., using np.random.randn with scaling).
    - `forward_pass(self, X: np.ndarray) -> tuple[np.ndarray, list]`: Compute activations layer-by-layer, return predictions and caches (A, Z for backprop). Handle output layer (e.g., softmax for multiclass).
    - `backward_pass(self, y_hat: np.ndarray, y_true: np.ndarray, caches: list) -> dict`: Compute gradients using chain rule (from "Gradients and initialization.pdf" and "Fitting models.pdf"). Sum over examples.
    - `update_params(self, grads: dict, learning_rate: float)`: Apply optimizer to update weights/biases.
    - `train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float, batch_size: int = None)`: Training loop with loss computation, backprop, updates. Support GD/SGD (mini-batches).
  - Helper functions: e.g., `compute_loss(self, y_hat, y_true)` (delegate to losses.py).
  - **Documentation Tips**: Docstrings for each method (e.g., explain backprop math with references to PDFs). Comments on vectorized NumPy ops for efficiency.

#### 2. `src/activations.py`

- **Purpose**: Modular activation functions and their derivatives (from "Shallow neural networks.pdf" Page 1-2). Supports ReLU, sigmoid, softmax (for output).
- **Key Contents**:
  - `def relu(z: np.ndarray) -> np.ndarray`: np.maximum(0, z)
  - `def relu_derivative(z: np.ndarray) -> np.ndarray`: (z > 0).astype(float)
  - `def sigmoid(z: np.ndarray) -> np.ndarray`: 1 / (1 + np.exp(-z))
  - `def sigmoid_derivative(z: np.ndarray) -> np.ndarray`: sigmoid(z) \* (1 - sigmoid(z))
  - `def softmax(z: np.ndarray) -> np.ndarray`: Stable version with axis=1, subtract max.
  - `def get_activation(name: str) -> callable`: Factory function to return activation/derivative pair.
  - **Documentation Tips**: Docstrings with math formulas (e.g., from PDFs). Comments on why ReLU is default (non-linearity from "Shallow neural networks.pdf").

#### 3. `src/losses.py`

- **Purpose**: Loss functions and derivatives for regression/classification (from "Loss functions.pdf" Page 1-2 and "Fitting models.pdf").
- **Key Contents**:
  - `def mse(y_hat: np.ndarray, y_true: np.ndarray) -> float`: Mean squared error (1/N \* sum((y_hat - y_true)^2))
  - `def mse_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray`: 2 \* (y_hat - y_true) / N
  - `def binary_cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> float`: -1/N _ sum(y_true _ log(y_hat) + (1-y_true) \* log(1-y_hat))
  - `def bce_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray`: (y_hat - y_true) / (y_hat \* (1 - y_hat))
  - `def categorical_cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> float`: -1/N _ sum(y_true _ log(y_hat)) (one-hot y_true)
  - `def cce_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray`: y_hat - y_true
  - `def get_loss(name: str) -> tuple[callable, callable]`: Factory for loss and derivative.
  - **Documentation Tips**: Reference maximum likelihood from "Loss functions.pdf". Comments on handling numerical stability (e.g., clip log inputs).

#### 4. `src/optimizers.py`

- **Purpose**: Optimization algorithms for parameter updates (from your plan's Step 6, inspired by "Fitting models.pdf").
- **Key Contents**:
  - `class Optimizer`: Base class with `update(self, params: dict, grads: dict, learning_rate: float)`.
  - `class GradientDescent(Optimizer)`: Basic update: params -= learning_rate \* grads
  - `class Momentum(Optimizer)`: Velocity update with beta (e.g., v = beta _ v + (1-beta) _ grads)
  - `class Adam(Optimizer)`: First/second moment estimates, bias correction, adaptive LR.
  - `def get_optimizer(name: str, **kwargs) -> Optimizer`: Factory with params like beta1, beta2, epsilon.
  - **Documentation Tips**: Docstrings with formulas. Comments on hyperparameters (e.g., "Beta from momentum in plan").

#### 5. `notebooks/train.ipynb`

- **Purpose**: Demonstrates the full pipeline (data prep, training, evaluation, visualizations). Use for experiments.
- **Key Sections** (as notebook cells):
  - Import modules from src/.
  - Data Preparation: Generate dataset (e.g., sklearn.make_moons for binary classification), normalize, split train/test (80/20), visualize (matplotlib scatter/line plot).
  - Model Instantiation: e.g., nn = NeuralNetwork(layer_dims=[2, 16, 8, 1], activations=['relu', 'relu', 'sigmoid'], loss_type='bce', optimizer_type='adam')
  - Training: Call nn.train(), plot loss vs. epochs (train/test).
  - Evaluation: Compute accuracy/MSE on test set, plot decision boundaries (for 2D data), compare train vs. test error.
  - Hyperparameter Search (Optional): Loop over LR/hidden units, plot results.
  - Extensions (Optional): Add batch norm, dropout, etc.
  - **Documentation Tips**: Markdown cells explaining each step (reference PDFs, e.g., "Overfitting from 'Supervised Learning.pdf' Page 2"). Inline comments in code cells.

#### 6. `data/` (Optional)

- **Purpose**: Store generated/processed datasets.
- **Contents**: CSV files or NumPy arrays (e.g., X_train.npy, y_train.npy). Generate in notebook if small.

#### 7. `docs/architecture.md`

- **Purpose**: Parallel/proper documentation of the project (e.g., high-level overviews, diagrams).
- **Key Contents**:
  - Project overview and goal (copy from your plan).
  - Diagrams: Neural net architecture (e.g., ASCII or link to Draw.io), forward/backprop flow (inspired by "Deep neural networks.pdf" and "Gradients and initialization.pdf").
  - Explanations: Link concepts to PDFs (e.g., "Loss functions from 'Loss functions.pdf' using MLE").
  - Extensions roadmap.
  - **Tips**: Update as you code for "parallel" documentation.

#### 8. `README.md`

- **Purpose**: Entry point for the project.
- **Key Contents**:
  - Project title/goal.
  - Setup: `pip install -r requirements.txt`
  - Usage: Run `train.ipynb` for demo.
  - Dependencies: numpy==1.26.4, matplotlib==3.9.2, scikit-learn==1.5.2 (for data gen only).
  - Estimated time: 15-20 hours (from plan).

#### 9. `requirements.txt`

- **Contents**:
  ```
  numpy
  matplotlib
  scikit-learn  # Only for data generation
  ```

### Additional Guidelines

- **Testing**: Add unit tests in src/ (e.g., test forward_pass with toy data). Use np.allclose for gradients.
- **Vectorization**: Ensure all ops are vectorized with NumPy (no loops over examples/layers where possible) for efficiency.
- **Extensions**: Implement in separate branches or optional methods (e.g., add dropout to NeuralNetwork).
- **References to PDFs**: Use them for math validation (e.g., gradient descent from "Fitting models.pdf", depth advantages from "Deep neural networks.pdf").
- **Version Control**: Use Git; commit often with messages like "Implement backprop per plan Step 5".

This template keeps things organized and scalable. Start with data prep in the notebook, then build the core class. If you need adjustments, let me know!
