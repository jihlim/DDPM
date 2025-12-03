import os

import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

def get_input_image_size_and_channels(dataset_name):
    if dataset_name.lower() == "mnist":
        i_size = 32
        channels = 1
    elif dataset_name.lower() == "cifar10":
        i_size = 32
        channels = 3
    elif dataset_name.lower() == "flowers102":
        i_size = 32
        channels = 3
    return i_size, channels

def get_dataset(data_root, dataset_name, i_size):
    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((i_size, i_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((i_size, i_size)),
            transforms.ToTensor()
        ]
    )

    # Datasets
    if dataset_name.lower() == "mnist":
        trainset = datasets.MNIST(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        testset = datasets.MNIST(
            root=data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        
    elif dataset_name.lower() == "cifar10":
        trainset = datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        testset = datasets.CIFAR10(
            root=data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        
    elif dataset_name.lower() == "flowers102":
        trainset = datasets.Flowers102(
            root=data_root,
            split="train",
            transform=train_transform,
            download=True
        )
        testset = datasets.Flowers102(
            root=data_root,
            split="val",
            transform=test_transform,
            download=True
        )

    return trainset, testset

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    print(root_dir)