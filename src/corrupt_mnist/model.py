import torch
from torch import nn


class Model(nn.Module):
    """
    Convolutional Neural Network with 3 convolutional layers,
    one fully connected layer, max_pooling, and relu activation functions.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 1, 28, 28), where N is the batch size.

        Returns:
            Tensor of shape (N, 10) with class logits.
        """
        # After conv1 (kernel=3, stride=1, padding=0): (N, 32, 26, 26)
        x = torch.relu(self.conv1(x))
        # After max pool (kernel=2, stride=2): (N, 32, 13, 13)
        x = torch.max_pool2d(x, 2, 2)
        # After conv2 (kernel=3, stride=1, padding=0): (N, 64, 11, 11)
        x = torch.relu(self.conv2(x))
        # After max pool (kernel=2, stride=2): (N, 64, 5, 5)
        x = torch.max_pool2d(x, 2, 2)
        # After conv3 (kernel=3, stride=1, padding=0): (N, 128, 3, 3)
        x = torch.relu(self.conv3(x))
        # After max pool (kernel=2, stride=2): (N, 128, 1, 1)
        x = torch.max_pool2d(x, 2, 2)
        # After flatten along channels and spatial dims: (N, 128)
        x = torch.flatten(x, 1)
        # After dropout: (N, 128)
        x = self.dropout(x)
        # Final linear layer to logits: (N, 10)
        return self.fc1(x)


if __name__ == "__main__":
    model = Model()
    print(f"Model Architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    x = torch.rand(1, 1, 28, 28)
    print(f"Output shape of model: {model(x).shape}")
