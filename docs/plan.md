1. Setup project structure: Create directories (src/, notebooks/) and files (neural_network.py, activations.py, losses.py, optimizers.py, train.ipynb).

2. Code data preparation: In train.ipynb, generate dataset (e.g., make_moons), normalize, split train/test, visualize.

3. Implement activations: In activations.py, code ReLU, sigmoid, softmax and derivatives; add factory function.

4. Implement losses: In losses.py, code MSE, BCE, CCE and derivatives; add factory.

5. Build model class: In neural_network.py, add **init** with layer_dims, activations, loss_type, optimizer_type; implement \_initialize_params (He/Xavier).

6. Code forward pass: In neural_network.py, add forward_pass method to compute layer activations, caches.

7. Code backpropagation: In neural_network.py, add backward_pass to compute gradients using chain rule.

8. Implement optimizers: In optimizers.py, code GD, Momentum, Adam classes; integrate into model.

9. Add training loop: In neural_network.py, implement train method with epochs, forward, loss, backward, updates; support mini-batches.

10. Add evaluation: In neural_network.py, add predict, compute accuracy/MSE; in train.ipynb, train model, plot losses, boundaries, analyze over/underfitting.

11. Extensions (optional): Add batch norm, dropout, LR scheduling in neural_network.py; test in notebook.
