def warmup(iteration, optimizer, target_lr, warmup_period):
    if iteration >= warmup_period:
        return optimizer.param_groups[0]['lr']

    lr = (0.5 * target_lr * (iteration + 1) / warmup_period) + 0.5 * target_lr
    change_lr(optimizer, lr)
    return lr

def change_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr