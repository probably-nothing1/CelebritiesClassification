from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, \
                                    CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts


def warmup(iteration, optimizer, target_lr, warmup_period):
    if iteration >= warmup_period:
        return
    lr = (0.5 * target_lr * (iteration + 1) / warmup_period) + 0.5 * target_lr
    change_lr(optimizer, lr)

def change_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

class IdleScheduler(object):
    def __init__(self):
        pass

    def step(self):
        pass

def dispatch_lr_scheduler(optimizer, args):
    if args.lr_scheduler is None:
        return IdleScheduler()
    elif args.lr_scheduler == 'StepLR':
        return StepLR(optimizer, step_size=args.step_lr_step_size, gamma=args.step_lr_gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        return MultiStepLR(optimizer, milestones=args.multistep_lr_milestones, gamma=args.multistep_lr_gamma)
    elif args.lr_scheduler == 'CyclicLR':
        return CyclicLR(optimizer, base_lr=args.learning_rate, gamma=args.cyclic_lr_gamma)
    elif args.lr_scheduler == 'OneCycleLR':
        return OneCycleLR(optimizer)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        return CosineAnnealingLR()
    elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        return CosineAnnealingWarmRestarts(optimizer)
