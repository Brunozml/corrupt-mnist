"""

Assuming open access, test locally with:
    docker run -v ~/.config/gcloud:/root/.config/gcloud:ro \
  -e GOOGLE_CLOUD_PROJECT=corrupt-mnist-26 \
  train:latest

  with local keyfile:
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/keyfile.json
  then run:
  docker run \
  -v $GOOGLE_APPLICATION_CREDENTIALS:/root/gcp-key.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/root/gcp-key.json \
  -e GOOGLE_CLOUD_PROJECT=corrupt-mnist-26 \
  train:latest
"""


from pathlib import Path

import torch
from omegaconf import OmegaConf

from corrupt_mnist.data import corrupt_mnist
from corrupt_mnist.model import Model

model_config = OmegaConf.load("configs/model_conf.yaml")
training_config = OmegaConf.load("configs/training_conf.yaml")
config = OmegaConf.merge(model_config, training_config)


# specify training device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Hyperparameters
def train(
    lr=config["hyperparameters"]["lr"],  # type: ignore[index]  # OmegaConf DictConfig typing
    batch_size=config["hyperparameters"]["batch_size"],  # type: ignore[index]
    epochs=config["hyperparameters"]["epochs"],  # type: ignore[index]
    models_path: str = "models/",
    seed=config["hyperparameters"]["seed"],  # type: ignore[index]
) -> None:
    """Train a CNN model on the corrupt MNIST dataset.

    This function implements a standard supervised learning training loop:
    1. Loads the corrupt MNIST dataset
    2. Initializes the model and moves it to the appropriate device (CPU/GPU/MPS)
    3. Sets up Adam optimizer and CrossEntropy loss
    4. Trains for the specified number of epochs
    5. Saves the trained model weights to disk

    The training process follows the standard PyTorch training pattern:
    - Forward pass: compute predictions
    - Compute loss: compare predictions to ground truth
    - Backward pass: compute gradients via backpropagation
    - Optimizer step: update model weights

    Args:
        lr (float, optional): Learning rate for the Adam optimizer.
            Defaults to value from config file.
        batch_size (int, optional): Number of samples per training batch.
            Larger batches train faster but use more memory.
            Defaults to value from config file.
        epochs (int, optional): Number of complete passes through the training dataset.
            Defaults to value from config file.
        models_path (str, optional): Directory path where the trained model will be saved.
            Directory is created if it doesn't exist. Defaults to "models/".
        seed (int, optional): Random seed for reproducibility. If provided, sets
            torch.manual_seed() to ensure deterministic training runs.
            Defaults to value from config file.

    Returns:
        None. Side effects include:
            - Prints training progress (hyperparameters, loss every 100 iterations)
            - Saves model state dict to {models_path}/model.pth
            - Creates models_path directory if it doesn't exist

    Notes:
        - Uses the global DEVICE constant to determine hardware acceleration
        - Training statistics (loss, accuracy) are tracked but not returned
        - Model is saved in PyTorch's state_dict format (.pth file)
        - For reproducibility in MLOps pipelines, always set a seed value

    Example:
        >>> # Train with default config values
        >>> train()
        >>>
        >>> # Train with custom hyperparameters
        >>> train(lr=0.001, batch_size=32, epochs=10, seed=42)
    """
    # print out hyperparameters
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")

    if seed:
        # Set all seeds for reproducibility
        print(f"Seeded at {seed=}")
        torch.manual_seed(seed)

    dataset, _ = corrupt_mnist()
    model = Model().to(DEVICE)

    # setup
    Path(models_path).mkdir(parents=True, exist_ok=True)

    # instantiate data loader object
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # start optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()

    # record training statistics for later use
    statistics: dict[str, list[float]] = {"train_loss": [], "train_accuracy": []}

    # outer epoch loop
    for epoch in range(epochs):
        # inner optimization loop
        for i, (img, target) in enumerate(dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            # optimizer - reset gradients
            optimizer.zero_grad()

            # forward pass on current state of the model
            out = model(img)

            # calculate loss
            loss = loss_function(out, target)

            # loss - calculate gradients
            loss.backward()

            # optimizer - take optimization step
            optimizer.step()

            # calculate training statistics
            statistics["train_loss"].append(loss.item())
            accuracy = (out.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    # save model
    torch.save(model.state_dict(), f"{models_path}/model.pth")
    print(f"Training complete! model saved to {models_path}/model.pth")


def overfit_test(
    batch_size: int = 64,
    n_batches: int = 1,
    epochs: int = 500,
    lr: float = 0.01,
    models_path: str = "models/",
    seed: int = 1234,
) -> dict:
    """
    Test if model can overfit to a small batch of data.

    This validates that the model architecture and training loop can
    actually learn, and catches issues with the model early.

    Args:
        n_batches: Number of batches to overfit on (typically 1-2).
        epochs: Number of epochs to train on the small batch.
        lr: Learning rate.
        models_path: Directory to save checkpoint.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with training statistics.
    """
    if seed:
        torch.manual_seed(seed)

    dataset, _ = corrupt_mnist()
    model = Model().to(DEVICE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Get only n_batches
    batches = [batch for i, batch in enumerate(dataloader) if i < n_batches]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()

    statistics: dict[str, list[float]] = {"train_loss": [], "train_accuracy": []}

    for epoch in range(epochs):
        for img, target in batches:
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            out = model(img)
            loss = loss_function(out, target)
            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())
            accuracy = (out.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {statistics['train_loss'][-1]:.4f}")

    torch.save(model.state_dict(), f"{models_path}/overfit_test_checkpoint.pth")
    print(f"Overfit test complete! Final loss: {statistics['train_loss'][-1]:.4f}")

    return statistics


if __name__ == "__main__":
    import sys

    # to run overfit test: uv run python -m corrupt_mnist.train overfit-test
    if len(sys.argv) > 1 and sys.argv[1] == "overfit-test":
        overfit_test()
    else:
        train()
