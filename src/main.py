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

from models import SimpleModel
from metrics import compute_accuracy, compute_confusion_matrix
from dataset.constants import TRAIN_IMAGE_MEAN, TRAIN_IMAGE_STD, TEST_IMAGE_MEAN, TEST_IMAGE_STD

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for train/test script')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='training epochs')
    parser.add_argument('--train-batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='Testing batch size')
    parser.add_argument('--optimizer', choices=['SGD'], default='SGD')
    parser.add_argument('--data-dir', help='Path to data folders', required=True)
    parser.add_argument('--use-cpu', action='store_true')
    return parser.parse_args()


def compute_loss(model, dataloader, num=20):
    total_loss = 0
    model.eval()

    for i, (x, y) in enumerate(train_dataloader):
        if i > num:
            break
        x, y = x.to(device), y.to(device)
        y_raw_prediction, _ = model(x)
        loss = criterion(y_raw_prediction, y)
        total_loss += loss.item()

    model.train()
    return total_loss / num

if __name__ == '__main__':
    args = parse_args()
    use_cuda = not args.use_cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    train_transform = Compose([ToTensor(), Normalize(TRAIN_IMAGE_MEAN, TRAIN_IMAGE_STD)])
    test_transform = Compose([ToTensor(), Normalize(TEST_IMAGE_MEAN, TEST_IMAGE_STD)])
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train/'), transform=train_transform)
    test_dataset = ImageFolder(os.path.join(args.data_dir, 'test/'), transform=test_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=False)

    model = SimpleModel().to(device)

    wandb.init(project="classifying-celebrities", config=args)
    wandb.watch(model, log='all')
    config = wandb.config

    criterion = CrossEntropyLoss(reduction='mean')
    optimizer = SGD(model.parameters(), lr=args.learning_rate)

    training_accuracy = compute_accuracy(model, train_dataloader, device)
    test_accuracy = compute_accuracy(model, test_dataloader, device)
    wandb.log({'training accuracy': training_accuracy})
    wandb.log({'test_accuracy': test_accuracy})

    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            y_raw_prediction, _ = model(x)
            loss = criterion(y_raw_prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'training loss': loss})

            if i % 10 == 0:
                test_loss = compute_loss(model, test_dataloader)
                wandb.log({'test loss': loss})

        training_accuracy = compute_accuracy(model, train_dataloader, device)
        test_accuracy = compute_accuracy(model, test_dataloader, device)
        wandb.log({'training accuracy': training_accuracy})
        wandb.log({'test_accuracy': test_accuracy})