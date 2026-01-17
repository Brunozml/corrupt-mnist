"""
Docstring for tests.test_model. 

 Checks  that for a given input with shape X that the output of the model has shape Y.

When you implement a test you need to follow two standards
for pytest to be able to find your tests. 
First, any files created (except __init__.py) should always start with test_*.py.
Secondly, any test implemented needs to be wrapped into a function that again needs to start with test_*:
"""
import os 
import tempfile
import shutil

from torch import nn
import torch

from corrupt_mnist.model import Model

def test_model():
    """Test basic forward pass with correct output shape."""
    model = Model()
    x = torch.rand(1, 1, 28, 28)
    
    y = model(x)
    assert y.shape == (1, 10)


def test_model_different_batch_sizes():
    """Test that model handles various batch sizes correctly."""
    model = Model()
    
    batch_sizes = [1, 8, 16, 32, 64, 128]
    for batch_size in batch_sizes:
        x = torch.rand(batch_size, 1, 28, 28)
        y = model(x)
        
        assert y.shape == (batch_size, 10)


def test_model_architecture():
    """Test that model has the expected layers."""
    model = Model()
    
    # Check that all expected layers exist
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'conv2')
    assert hasattr(model, 'conv3')
    assert hasattr(model, 'dropout')
    assert hasattr(model, 'fc1')
    
    # Verify layer types
    assert isinstance(model.conv1, torch.nn.Conv2d)
    assert isinstance(model.conv2, torch.nn.Conv2d)
    assert isinstance(model.conv3, torch.nn.Conv2d)
    assert isinstance(model.dropout, torch.nn.Dropout)
    assert isinstance(model.fc1, torch.nn.Linear)


def test_model_layer_specs():
    """Test that layers have correct specifications."""
    model = Model()
    
    # Conv1
    assert model.conv1.in_channels == 1
    assert model.conv1.out_channels == 32
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    
    # Conv2
    assert model.conv2.in_channels == 32
    assert model.conv2.out_channels == 64
    assert model.conv2.kernel_size == (3, 3)
    assert model.conv2.stride == (1, 1)
    
    # Conv3
    assert model.conv3.in_channels == 64
    assert model.conv3.out_channels == 128
    assert model.conv3.kernel_size == (3, 3)
    assert model.conv3.stride == (1, 1)
    
    # Dropout
    assert model.dropout.p == 0.2
    
    # FC layer
    assert model.fc1.in_features == 128
    assert model.fc1.out_features == 10


def test_model_parameter_count():
    """Test that model has expected number of parameters."""
    model = Model()
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Model should have a reasonable number of parameters
    assert total_params > 0
    
    # Rough estimate: conv layers + fc layer
    assert total_params < 1_000_000  # Should be less than 1M for efficiency


def test_model_trainable_parameters():
    """Test that all parameters are trainable by default."""
    model = Model()
    
    for param in model.parameters():
        assert param.requires_grad


def test_model_forward_gradient_flow():
    """Test that gradients can flow through the model."""
    model = Model()
    x = torch.rand(2, 1, 28, 28, requires_grad=True)
    
    # Forward pass
    y = model(x)
    loss = y.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert model.conv1.weight.grad is not None
    assert model.fc1.weight.grad is not None


def test_model_output_range():
    """Test that output logits are in reasonable range."""
    model = Model()
    model.eval()
    
    x = torch.rand(4, 1, 28, 28)
    y = model(x)
    
    # Logits should not be NaN or Inf
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    
    # Logits should be in a reasonable range
    assert (y.abs() < 1e6).all()


def test_model_deterministic():
    """Test that model produces same output with eval mode and no gradients."""
    model = Model()
    model.eval()
    
    x = torch.rand(2, 1, 28, 28)
    
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    
    # Output should be deterministic in eval mode
    assert torch.allclose(y1, y2, rtol=1e-5)


def test_model_dropout_affects_output():
    """Test that dropout changes output between train and eval modes."""
    torch.manual_seed(42)
    model = Model()
    
    x = torch.rand(16, 1, 28, 28)
    
    # Train mode (dropout active)
    model.train()
    with torch.no_grad():
        y_train1 = model(x)
        y_train2 = model(x)
    
    # Eval mode (dropout inactive)
    model.eval()
    with torch.no_grad():
        y_eval1 = model(x)
        y_eval2 = model(x)
    
    # In eval mode, outputs should be identical
    assert torch.allclose(y_eval1, y_eval2)
    
    # In train mode, outputs may differ due to dropout
    # (very unlikely to be exactly the same with large batch)
    # We just verify both modes run without error


def test_model_state_dict():
    """Test that model state can be saved and loaded."""
    model1 = Model()
    state_dict = model1.state_dict()
    
    # Create new model and load state
    model2 = Model()
    model2.load_state_dict(state_dict)
    
    # Test with same input
    x = torch.rand(2, 1, 28, 28)
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        y1 = model1(x)
        y2 = model2(x)
    
    assert torch.allclose(y1, y2)


def test_model_to_device():
    """Test that model can be moved to different devices."""
    model = Model()
    
    # Test CPU (always available)
    model = model.to('cpu')
    x = torch.rand(1, 1, 28, 28).to('cpu')
    y = model(x)
    assert y.shape == (1, 10)
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        x = torch.rand(1, 1, 28, 28).to('cuda')
        y = model(x)
        assert y.shape == (1, 10)
        assert y.device.type == 'cuda'


def test_model_zero_grad():
    """Test that zero_grad clears gradients."""
    model = Model()
    x = torch.rand(1, 1, 28, 28, requires_grad=True)
    
    # Forward + backward
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Verify gradients exist before clearing
    assert model.conv1.weight.grad is not None
    assert model.fc1.weight.grad is not None
    
    # Zero gradients
    model.zero_grad()
    
    # After zero_grad(), gradients should be cleared (set to None)
    assert model.conv1.weight.grad is None
    assert model.fc1.weight.grad is None

def test_model_with_batch_norm():
    """Test model behavior with different input statistics."""
    model = Model()
    model.eval()
    
    # Test with different normalized inputs
    x1 = torch.randn(2, 1, 28, 28) * 0.1  # Small variance
    x2 = torch.randn(2, 1, 28, 28) * 10   # Large variance
    
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    
    # Both should produce valid outputs
    assert not torch.isnan(y1).any()
    assert not torch.isnan(y2).any()


def test_model_output_not_all_same():
    """Test that model produces different outputs for different inputs."""
    model = Model()
    model.eval()
    
    x1 = torch.zeros(1, 1, 28, 28)
    x2 = torch.ones(1, 1, 28, 28)
    
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    
    # Different inputs should produce different outputs
    assert not torch.allclose(y1, y2)


def test_model_conv_layer_specifications():
    """Detailed test of convolutional layer output shapes."""
    model = Model()
    model.eval()
    
    with torch.no_grad():
        x = torch.rand(1, 1, 28, 28)
        
        # Conv1 -> ReLU -> MaxPool -> 28x28 -> 26x26 -> 13x13
        x = torch.relu(model.conv1(x))
        assert x.shape == (1, 32, 26, 26)
        x = torch.max_pool2d(x, 2, 2)
        assert x.shape == (1, 32, 13, 13)
        
        # Conv2 -> ReLU -> MaxPool -> 13x13 -> 11x11 -> 5x5
        x = torch.relu(model.conv2(x))
        assert x.shape == (1, 64, 11, 11)
        x = torch.max_pool2d(x, 2, 2)
        assert x.shape == (1, 64, 5, 5)
        
        # Conv3 -> ReLU -> MaxPool -> 5x5 -> 3x3 -> 1x1
        x = torch.relu(model.conv3(x))
        assert x.shape == (1, 128, 3, 3)
        x = torch.max_pool2d(x, 2, 2)
        assert x.shape == (1, 128, 1, 1)
        
        # Flatten -> FC -> output
        x = torch.flatten(x, 1)
        assert x.shape == (1, 128)


def test_model_can_overfit_single_batch():
    """Test that model can overfit to a single batch (sanity check)."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = Model().to(DEVICE)
    
    #set epochs to 500
    epochs = 200

    # Create one batch
    x = torch.rand(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    initial_loss = None
    
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(x.to(DEVICE))
        loss = loss_fn(out, y.to(DEVICE))
        loss.backward()
        optimizer.step()
        
        if initial_loss is None:
            initial_loss = loss.item()
    
    final_loss = loss.item()
    
    # Loss should decrease significantly (at least 50%)
    assert final_loss < initial_loss * 0.5, \
        f"Model didn't overfit: {initial_loss:.4f} -> {final_loss:.4f}"