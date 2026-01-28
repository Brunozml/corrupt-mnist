import tempfile

import torch

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
            # assert that each datapoint has shape [1,28,28]
            assert x.shape == (1, 28, 28)

            # assert that all labels are in range
            assert y in range(10)

    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all()


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


def test_corrupt_mnist_saves_files(tmp_path):
    """Test that corrupt_mnist saves processed files correctly."""
    # Create raw data directory with mock files
    raw_path = tmp_path / "raw"
    raw_path.mkdir()

    # Create dummy train files (6 files)
    for i in range(6):
        torch.save(torch.randn(100, 28, 28), raw_path / f"train_images_{i}.pt")
        torch.save(torch.randint(0, 10, (100,)), raw_path / f"train_target_{i}.pt")

    # Create dummy test files
    torch.save(torch.randn(50, 28, 28), raw_path / "test_images.pt")
    torch.save(torch.randint(0, 10, (50,)), raw_path / "test_target.pt")

    # Call corrupt_mnist
    train, test = corrupt_mnist(tmp_path)

    # Assert processed folder exists
    processed_path = tmp_path / "processed"
    assert processed_path.exists()

    # Assert all files are saved
    assert (processed_path / "train_images.pt").exists()
    assert (processed_path / "train_target.pt").exists()
    assert (processed_path / "test_images.pt").exists()
    assert (processed_path / "test_target.pt").exists()

    # Assert tensors have correct shape
    assert train.tensors[0].shape == (600, 1, 28, 28)  # 6 files * 100 samples
    assert test.tensors[0].shape == (50, 1, 28, 28)


if __name__ == "__main__":
    test_data()
    test_normalize()
    test_normalize_shape_preserved()
    test_corrupt_mnist_saves_files(tempfile.TemporaryDirectory().name)
    print("All tests passed!")
