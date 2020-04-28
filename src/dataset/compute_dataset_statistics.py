import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for computing statistics')
    parser.add_argument('--train-batch-size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1024, help='Testing batch size')
    parser.add_argument('--data-dir', help='Path to data folders', required=True)
    return parser.parse_args()

def compute_dataset_statistics(dataloader):
    means, stds = None, None
    for i, (x, _) in enumerate(dataloader):
        if i == 0:
            stds, means = torch.std_mean(x, dim=(0, 2, 3))
            stds, means = stds.unsqueeze(0), means.unsqueeze(0)
            continue

        std, mean = torch.std_mean(x, dim=(0, 2, 3))
        std, mean = std.unsqueeze(0), mean.unsqueeze(0)

        means = torch.cat((means, mean), dim=0)
        stds = torch.cat((stds, std), dim=0)

    means = means.mean(dim=0)
    stds = stds.mean(dim=0)
    return means, stds


if __name__ == '__main__':
    args = parse_args()

    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train/'), transform=ToTensor())
    test_dataset = ImageFolder(os.path.join(args.data_dir, 'test/'), transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False, drop_last=False)

    mean, std = compute_dataset_statistics(train_dataloader)
    print(f'Train dataset:\n\t\
            mean: {mean}\n\t\
            std: {std}')

    mean, std = compute_dataset_statistics(test_dataloader)
    print(f'Test dataset:\n\t\
            mean: {mean}\n\t\
            std: {std}' )
