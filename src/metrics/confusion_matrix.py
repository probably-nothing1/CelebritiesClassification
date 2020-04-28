import torch
import matplotlib.pyplot as plt
import seaborn

def compute_confusion_matrix(model, dataloader, device, num_classes=28):
    model.eval()
    confusion_matrix = torch.zeros((num_classes, num_classes))
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        _, probabilities = model(x)
        predicted_classes = torch.argmax(probabilities, dim=1)
        for y_gt, y_pred in zip(y, predicted_classes):
            confusion_matrix[y_gt, y_pred] += 1

    model.train()
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, num_classes=28):
    import matplotlib.pyplot as plt
    import seaborn
    fig, ax = plt.subplots(figsize=(20,15))
    seaborn.heatmap(confusion_matrix, annot=True, xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
