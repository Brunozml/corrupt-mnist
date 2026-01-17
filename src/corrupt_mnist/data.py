from pathlib import Path

import torch
import typer
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting


# class MyDataset(Dataset):
#     """My custom dataset."""

#     def __init__(self, data_path: Path) -> None:
#         self.data_path = data_path

#     def __len__(self) -> int:
#         """Return the length of the dataset."""
#         ...

#     def __getitem__(self, index: int):
#         """Return a given sample from the dataset."""



# def preprocess(data_path: Path, output_folder: Path) -> None:
#     print("Preprocessing data...")
#     dataset = MyDataset(data_path)
#     dataset.preprocess(output_folder)


def normalize(images):
    """ Normalize images to have zero mean and unit variance """
    return (images - images.mean()) / images.std()

def corrupt_mnist(data_path: Path = 'data'):
    """Return train and test datasets for corrupt MNIST."""
    data_path = Path(data_path)
    
    # load train data
    train_images, train_targets = [], []
    for i in range(6):
        images_path = data_path / 'raw' / f'train_images_{i}.pt'
        targets_path = data_path / 'raw' / f'train_target_{i}.pt'
        if not images_path.exists():
            raise FileNotFoundError(f"Train data not found at {images_path}")
        train_images.append(torch.load(images_path))
        if not targets_path.exists():
            raise FileNotFoundError(f"Train targets not found at {targets_path}")
        train_targets.append(torch.load(targets_path))

    # concat along the first dimension to turn from list of tensors to a single tensor
    train_images = torch.cat(train_images, dim=0) # shape: [N_samples, width, height]
    train_targets = torch.cat(train_targets, dim=0) # shape: [N_samples]

    # load test data
    images_path = data_path / 'raw' / 'test_images.pt'
    targets_path = data_path / 'raw' / 'test_target.pt'
    if not images_path.exists():
        raise FileNotFoundError(f"Test data not found at {images_path}")
    test_images = torch.load(images_path)
    if not targets_path.exists():
        raise FileNotFoundError(f"Test targets not found at {targets_path}")
    test_targets = torch.load(targets_path)

    # normalize images
    train_images = normalize(train_images) # shape: [N_samples, width, height]
    test_images = normalize(test_images) # shape: [N_samples, width, height]
    
    # unsqueeze images from [N_samples , width, height] to [N_samples, 1, width, height]
    train_images = train_images.unsqueeze(dim=1) 
    test_images = test_images.unsqueeze(dim=1)

    # convert target to 64-bit int dtype (required for certain loss functions)
    train_targets = train_targets.long()
    test_targets = test_targets.long()

    # save to 'processed' folder
    processed_path = data_path / 'processed'
    processed_path.mkdir(parents=True, exist_ok=True)
    torch.save(train_images, processed_path / 'train_images.pt')
    torch.save(train_targets, processed_path / 'train_target.pt')
    torch.save(test_images, processed_path / 'test_images.pt')
    torch.save(test_targets, processed_path / 'test_target.pt')

        # convert to tensor dataset 
    train = torch.utils.data.TensorDataset(train_images, train_targets)
    test = torch.utils.data.TensorDataset(test_images, test_targets)

    return train, test

def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    train, test = corrupt_mnist()
    print(f"Size of training set: {len(train)}")
    print(f"Size of test set: {len(test)}")
    print(f"Shape of a training point {(train[0][0].shape, train[0][1].shape)}")
    print(f"Shape of a test point {(test[0][0].shape, test[0][1].shape)}")
    show_image_and_target(train.tensors[0][:25], train.tensors[1][:25])