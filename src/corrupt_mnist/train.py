import os

from corrupt_mnist.model import Model
from corrupt_mnist.data import corrupt_mnist
import torch
from omegaconf import OmegaConf
# loading

config = OmegaConf.load('config.yaml')


# specify training device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
def train(lr = config['hyperparameters']['lr'],
          batch_size = config['hyperparameters']['batch_size'],
          epochs = config['hyperparameters']['epochs'],
          models_path = "models/",
          seed = config['hyperparameters']['seed']):
    
    # print out hyperparameters
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")


    if seed: 
        # Set all seeds for reproducibility
        print(f'Seeded at {seed=}')
        torch.manual_seed(seed)

    dataset, _ = corrupt_mnist()
    model = Model().to(DEVICE)

    # setup
    os.makedirs(models_path, exist_ok=True)

    # instantiate data loader object 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # start optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()

    # record training statistics for later use
    statistics = {
        "train_loss": [],
        "train_accuracy": []
    }

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
    seed: int = 1234
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
    
    statistics = {"train_loss": [], "train_accuracy": []}
    
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
