def compute_loss(model, dataloader, loss_function, device, num=20):
    total_loss = 0
    model.eval()

    for i, (x, y) in enumerate(dataloader):
        if i > num:
            break
        x, y = x.to(device), y.to(device)
        y_raw_prediction, _ = model(x)
        loss = loss_function(y_raw_prediction, y)
        total_loss += loss.item()

    model.train()
    return total_loss / num