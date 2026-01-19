
import torch
import typer
from data import corrupt_mnist
from model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate_model(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    # set model to evaluation mode
    _, test_set = corrupt_mnist()

    # instantiate data loader object
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    # evaluation setup
    model.eval()
    correct, total = 0, 0
    # for each datapoint, split images and labels
    for img, target in dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)

        # forward pass on the test set
        y_pred = model(img)

        # calculate running number of correct and total predictions
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.shape[0]
    
    # calculate accuracy
    accuracy = correct / total
    print(f"Test set accuracy: {accuracy:.4f}")
    print("Total samples:", total)
    print("Correct predictions:", correct)


if __name__ == "__main__":
    typer.run(evaluate_model)