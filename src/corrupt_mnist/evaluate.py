from pathlib import Path
from typing import Annotated
import torch
import typer

from data import corrupt_mnist
from model import Model

app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@app.command()
def evaluate(
    model_checkpoint: Annotated[str, typer.Option("--model-checkpoint", "-model")],
    data_path: Annotated[str, typer.Option("--data-path", "-data")] = 'data'
    ) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print("Model checkpoint:", model_checkpoint)
    print("Data path:" , data_path)

    # load the model (note: for docker build using CPI explicitly)
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    # set model to evaluation mode
    _, test_set = corrupt_mnist(Path(data_path))

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
    app()