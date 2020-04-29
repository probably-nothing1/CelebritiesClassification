import argparse
import os
import wandb

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from models import SimpleModel
from training import warmup, dispatch_lr_scheduler, get_lr, dispatch_optimizer
from metrics import compute_accuracy, compute_confusion_matrix, compute_loss
from dataset import get_train_dataloader, get_test_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for train/test script')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='training epochs')
    parser.add_argument('--train-batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=32, help='Testing batch size')
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW'], default='SGD')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--lr-scheduler', default=None, choices=[None, 'StepLR', 'MultiStepLR', 'CosineAnnealingLR', 'CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts'])
    parser.add_argument('--step-lr-step-size', type=int, default=1)
    parser.add_argument('--step-lr-gamma', type=float, default=0.9)
    parser.add_argument('--multistep-lr-milestones', nargs='+', type=int, default=[5, 10])
    parser.add_argument('--multistep-lr-gamma', type=float, default=0.3)
    parser.add_argument('--data-dir', help='Path to data folders', required=True)
    parser.add_argument('--use-cpu', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    use_cuda = not args.use_cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    train_dataloader = get_train_dataloader(os.path.join(args.data_dir, 'train/'), args.train_batch_size, args.augmentation)
    test_dataloader = get_test_dataloader(os.path.join(args.data_dir, 'test/'), args.test_batch_size)

    model = SimpleModel().to(device)

    wandb.init(project="classifying-celebrities", config=args)
    wandb.watch(model, log='all')
    config = wandb.config

    loss_function = CrossEntropyLoss(reduction='mean')
    optimizer = dispatch_optimizer(model, args)
    lr_scheduler = dispatch_lr_scheduler(optimizer, args)

    iteration = 0
    training_accuracy = compute_accuracy(model, train_dataloader, device)
    test_accuracy = compute_accuracy(model, test_dataloader, device)
    wandb.log({'training accuracy': training_accuracy}, step=iteration)
    wandb.log({'test_accuracy': test_accuracy}, step=iteration)

    for epoch in range(args.epochs):
        for x, y in train_dataloader:
            if iteration < args.warmup:
                warmup(iteration, optimizer, args.learning_rate, args.warmup)
            x, y = x.to(device), y.to(device)
            y_raw_prediction, _ = model(x)
            loss = loss_function(y_raw_prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'training loss': loss}, step=iteration)
            wandb.log({'learning rate': get_lr(optimizer)}, step=iteration)

            if iteration % 10 == 0:
                test_loss = compute_loss(model, test_dataloader, loss_function, device)
                wandb.log({'test loss': loss}, step=iteration)
            wandb.log({'iteration': iteration}, step=iteration)
            iteration += 1

        lr_scheduler.step()
        training_accuracy = compute_accuracy(model, train_dataloader, device)
        test_accuracy = compute_accuracy(model, test_dataloader, device)
        wandb.log({'training accuracy': training_accuracy}, step=iteration)
        wandb.log({'test_accuracy': test_accuracy}, step=iteration)