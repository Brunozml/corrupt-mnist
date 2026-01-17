import os
import tempfile
import shutil 

from torch.utils.data import Dataset
import torch
import pytest

from corrupt_mnist.data import corrupt_mnist, normalize

def test_data():
    """Test corrupt_mnist function for correct dataset sizes and shapes."""
    train, test = corrupt_mnist()

    #  assert len(dataset) == N_train for training and N_test for test
    assert len(train) == 30_000
    assert len(test) == 5_000
    # checks on each individual datapoint
    for dataset in [train, test]:
        for x, y in dataset:
            #assert that each datapoint has shape [1,28,28]
            assert x.shape == (1, 28, 28)

             # assert that all labels are in range
            assert y in range(10)
    
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()


def test_normalize():
    """Test normalize function for correct mean and std."""
    # Create test tensor with known values
    test_images = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    normalized = normalize(test_images)
    
    # After normalization, mean should be ~0 and std should be ~1
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalized.std(), torch.tensor(1.0), atol=1e-6)


def test_normalize_shape_preserved():
    """Test that normalize preserves tensor shape."""
    shapes = [(10, 28, 28), (100, 1, 28, 28), (5,)]
    
    for shape in shapes:
        test_tensor = torch.randn(shape)
        normalized = normalize(test_tensor)
        assert normalized.shape == test_tensor.shape



if __name__ == "__main__":
    test_data()
    test_normalize()
    test_normalize_shape_preserved()
    print("All tests passed!")