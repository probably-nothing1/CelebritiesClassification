import torch

def compute_accuracy(model, dataloader, device):
    num_correct = 0
    num_images = len(dataloader) * dataloader.batch_size
    model.eval()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        _, probabilities = model(x)
        predicted_classes = torch.argmax(probabilities, dim=1)
        num_correct += (predicted_classes == y).sum().item()

    model.train()
    return num_correct / num_images