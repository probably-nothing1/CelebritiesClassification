import argparse
import os
import time

import wandb
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from models import SimpleModel
from training import warmup, dispatch_lr_scheduler, get_lr, dispatch_optimizer
from metrics import compute_accuracy, compute_confusion_matrix, compute_loss
from dataset import get_train_dataloader, get_test_dataloader
from utils import parse_args


if __name__ == '__main__':
    args = parse_args()
    use_cuda = not args.use_cpu and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    bs = args.train_batch_size

    train_dataloader = get_train_dataloader(os.path.join(args.data_dir, 'train/'), args.train_batch_size, args.augmentation)
    test_dataloader = get_test_dataloader(os.path.join(args.data_dir, 'test/'), args.test_batch_size)

    model = SimpleModel(use_bn=args.use_bn).to(device)

    wandb.init(project="classifying-celebrities", config=args)
    wandb.watch(model, log='all')
    config = wandb.config

    loss_function = CrossEntropyLoss(reduction='mean')
    optimizer = dispatch_optimizer(model, args)
    lr_scheduler = dispatch_lr_scheduler(optimizer, args)

    iteration = 0
    training_accuracy = compute_accuracy(model, train_dataloader, device)
    test_accuracy = compute_accuracy(model, test_dataloader, device)
    wandb.log({'training accuracy': training_accuracy}, step=iteration*bs)
    wandb.log({'test_accuracy': test_accuracy}, step=iteration*bs)

    for epoch in range(args.epochs):
        for x, y in train_dataloader:
            start_time = time.time()
            if iteration < args.warmup:
                warmup(iteration, optimizer, args.learning_rate, args.warmup)
            x, y = x.to(device), y.to(device)
            y_raw_prediction, _ = model(x)
            loss = loss_function(y_raw_prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'training loss': loss}, step=iteration*bs)
            wandb.log({'learning rate': get_lr(optimizer)}, step=iteration*bs)

            if iteration % 10 == 0:
                test_loss = compute_loss(model, test_dataloader, loss_function, device)
                wandb.log({'test loss': loss}, step=iteration*bs)
            wandb.log({'iteration': iteration}, step=iteration * bs)
            wandb.log({'iteration time': time.time() - start_time}, step=iteration*bs)
            iteration += 1

        lr_scheduler.step()
        training_accuracy = compute_accuracy(model, train_dataloader, device)
        test_accuracy = compute_accuracy(model, test_dataloader, device)
        wandb.log({'training accuracy': training_accuracy}, step=iteration*bs)
        wandb.log({'test_accuracy': test_accuracy}, step=iteration*bs)