from torch.optim import SGD, Adam, AdamW, RMSprop, Adagrad

def dispatch_optimizer(model, args):
    if args.optimizer == 'SGD':
        return SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == 'Adam':
        return Adam(model.parameters(), lr=args.learning_rate)
    if args.optimizer == 'AdamW':
        return AdamW(model.parameters(), lr=args.learning_rate)
    if args.optimizer == 'RMSprop':
        return RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == 'Adagrad':
        return Adagrad(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
