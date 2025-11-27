# Neural Network from Scratch

![Neural Network](https://img.shields.io/badge/Neural%20Network-From%20Scratch-blueviolet?style=flat-square&logo=python)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A lightweight, educational implementation of a fully-connected feedforward neural network built entirely from scratch using only NumPy. This project is designed for learning purposes, demonstrating core concepts like forward/backward propagation, various activations, losses, optimizers, and metrics without relying on high-level libraries like TensorFlow or PyTorch.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Flexible Architecture**: Define arbitrary layer dimensions and activations (ReLU, Sigmoid, Softmax, Linear).
- **Initialization Strategies**: He (for ReLU) and Xavier (for Sigmoid/Softmax/Linear) weight initialization.
- **Loss Functions**: Mean Squared Error (MSE), Binary Cross-Entropy (BCE), Categorical Cross-Entropy (CCE).
- **Optimizers**: Gradient Descent (GD), Momentum, Adam.
- **Metrics**: Accuracy, Precision, Recall, F1-Score for classification; MAE, MSE, RMSE, R² for regression.
- **Data Generation**: Built-in utilities for creating toy datasets (e.g., Moons for classification, Noisy Sine for regression).
- **Training Support**: Mini-batch gradient descent with shuffling for better convergence.
- **Educational Focus**: Clean, documented code with type hints and docstrings for clarity.

This implementation supports binary/multiclass classification and regression tasks, making it a great starting point for understanding neural networks under the hood.

## Project Structure

```
neural-network-from-scratch/
├── src/
│   ├── activations.py      # Activation functions and derivatives (ReLU, Sigmoid, Softmax, Linear)
│   ├── generate_data.py    # Utilities for generating toy datasets (classification & regression)
│   ├── initialization.py   # Weight initialization strategies (He, Xavier)
│   ├── losses.py           # Loss functions and derivatives (MSE, BCE, CCE)
│   ├── metrics.py          # Evaluation metrics for classification and regression
│   ├── neural_network.py   # Core NeuralNetwork class (forward/backward pass, training)
│   └── optimizers.py       # Optimization algorithms (GD, Momentum, Adam)
├── notebooks/
│   └── [Jupyter notebooks with explanations, research, and visualizations for each module]
├── requirements.txt        # Dependency versions required for the project
└── README.md               # This file
```

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. Install dependencies using the provided requirements.txt:
   ```
   pip install -r requirements.txt
   ```

No additional setup is required—everything runs in pure Python!

## Usage

Import the `NeuralNetwork` class and use it to build, train, and evaluate models.

```python
import numpy as np
from src.neural_network import NeuralNetwork
from src.generate_data import get_classification_data
from src.metrics import get_classification_report, print_classification_report

# Generate toy data
X, y = get_classification_data(n_samples=1000, noise=0.1)

# Define and initialize the network
nn = NeuralNetwork(
    layer_dims=[2, 10, 1],  # Input: 2 features, Hidden: 10 units, Output: 1 unit
    activations=["relu", "sigmoid"],
    loss_type="bce",  # Binary Cross-Entropy for classification
    optimizer_type="adam",
    optimizer_params={"beta": 0.9, "hamma": 0.999},
    seed=42
)

# Train the model
losses = nn.train(X, y, epochs=100, learning_rate=0.01, batch_size=32)

# Predict and evaluate
y_hat, _ = nn.forward_pass(X)
y_hat_binary = (y_hat > 0.5).astype(int)  # Threshold for binary classification
acc, recall, precision, f1 = get_classification_report(y_hat_binary.flatten(), y.flatten(), num_classes=2)
print_classification_report(acc, recall, precision, f1, num_classes=2)
```

For regression tasks, swap to `get_regression_data`, use `"mse"` loss, and `"linear"` output activation.

## Contributing

Contributions are welcome! Feel free to open issues for bugs or feature requests. Pull requests should include tests and follow PEP8 style.

1. Fork the repo.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m 'Add feature'`.
4. Push: `git push origin feature/your-feature`.
5. Open a PR.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Thanks!
