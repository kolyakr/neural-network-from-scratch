import numpy as np
import pytest
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your functions
# Assumes your code is in src/activations.py
from src.activations import relu, relu_derivative, sigmoid, sigmoid_derivative, softmax, get_activation

class TestActivations:
    
    def test_relu_values(self):
        """Test that ReLU clips negative values to 0 and keeps positive ones."""
        z = np.array([[-1.0, 2.0], [0.0, -5.0]])
        expected = np.array([[0.0, 2.0], [0.0, 0.0]])
        
        output = relu(z)
        np.testing.assert_array_equal(output, expected)

    def test_relu_derivative(self):
        """Test ReLU derivative: 0 for z<=0, 1 for z>0."""
        z = np.array([[-2.0, 3.0], [0.0, 5.0]])
        # Note: strictly 0 is undefined, but usually treated as 0 or 1. 
        # Your implementation (z > 0) treats 0 as 0.
        expected = np.array([[0.0, 1.0], [0.0, 1.0]])
        
        output = relu_derivative(z)
        np.testing.assert_array_equal(output, expected)

    def test_sigmoid_values(self):
        """Test specific sigmoid properties (0 -> 0.5, large pos -> ~1)."""
        z = np.array([[0.0], [100.0], [-100.0]])
        output = sigmoid(z)
        
        assert output[0, 0] == 0.5
        assert output[1, 0] > 0.99  # Should be close to 1
        assert output[2, 0] < 0.01  # Should be close to 0

    def test_sigmoid_derivative(self):
        """Test derivative at z=0. Sigmoid(0)=0.5, so deriv = 0.5 * (1-0.5) = 0.25"""
        z = np.array([[0.0]])
        output = sigmoid_derivative(z)
        np.testing.assert_allclose(output, [[0.25]])

    def test_softmax_shapes_and_sum(self):
        """
        CRITICAL TEST for (Features, Examples) layout.
        Input: (3 Features, 5 Examples)
        Expected: Columns should sum to 1.
        """
        # Shape: (3 classes, 5 examples)
        z = np.random.randn(3, 5)
        output = softmax(z)
        
        # 1. Check Shape is preserved
        assert output.shape == (3, 5)
        
        # 2. Check Sums
        # We sum down axis 0 (features). Result should be shape (5,) all 1s.
        col_sums = np.sum(output, axis=0)
        
        # NOTE: This test will FAIL if your softmax uses axis=1
        np.testing.assert_allclose(col_sums, 1.0, err_msg="Softmax columns do not sum to 1")

    def test_get_activation_factory(self):
        """Test the factory function returns correct callables."""
        # Test ReLU
        func, deriv = get_activation("relu")
        assert func == relu
        assert deriv == relu_derivative
        
        # Test Softmax (Single return)
        func = get_activation("softmax")
        assert func == softmax
        
        # Test Invalid
        with pytest.raises(ValueError):
            get_activation("invalid_name")

if __name__ == "__main__":
    # Allows running this script directly with python test_activations.py
    pytest.main([__file__])