import argparse
import os
import wandb

import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

from model import SimpleModel

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for train/test script')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=3, help='epochs')
    parser.add_argument('--train-batch-size', '-train-bs', type=int, default=32, help='Training batch size')
    parser.add_argument('--test-batch-size' '-test-bs', type=int, default=32, help='Testing batch size')
    parser.add_argument('--optimizer', '-optim', choices=['SGD'], default='SGD')
    parser.add_argument('--data-dir', help='Path to data folders', required=True)
    parser.add_argument('--cuda', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    wandb.init(project="classifying-celebrities")
    args = parse_args()

    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train/'), transform=ToTensor())
    test_dataset = ImageFolder(os.path.join(args.data_dir, 'test/'), transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size, drop_last=True)

    model = SimpleModel()

    criterion = MSELoss(reduction='sum')
    optimizer = SGD(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(train_dataloader):
            y = F.one_hot(y, 28).float()
            y_prediction = model(x)
            loss = criterion(y_prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss})
