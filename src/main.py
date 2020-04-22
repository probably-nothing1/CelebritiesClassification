import argparse
import os
import wandb

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize

from model import SimpleModel
from utils.constants import TRAIN_IMAGE_MEAN, TRAIN_IMAGE_STD

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for train/test script')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='training epochs')
    parser.add_argument('--train-batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=32, help='Testing batch size')
    parser.add_argument('--optimizer', choices=['SGD'], default='SGD')
    parser.add_argument('--data-dir', help='Path to data folders', required=True)
    parser.add_argument('--cuda', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    use_cuda = not args.cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    train_transform = Compose([ToTensor(), Normalize(TRAIN_IMAGE_MEAN, TRAIN_IMAGE_STD)])
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train/'), transform=train_transform)
    test_dataset = ImageFolder(os.path.join(args.data_dir, 'test/'), transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)

    model = SimpleModel().to(device)

    wandb.init(project="classifying-celebrities", config=args)
    wandb.watch(model, log='all')
    config = wandb.config

    criterion = CrossEntropyLoss(reduction='mean')
    optimizer = SGD(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            y_prediction = model(x)
            loss = criterion(y_prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss})