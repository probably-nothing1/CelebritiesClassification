import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, num_classes=28, filter_sizes=[32, 64, 64, 128], use_bn=False, custom_batchnorm=False):
        super().__init__()
        self.num_classes = num_classes
        layers = []
        input_filter_sizes = [3] + filter_sizes[:-1]
        for input_size, output_size in zip(input_filter_sizes, filter_sizes):
            layers.append(nn.Conv2d(input_size, output_size, 7))
            if use_bn:
                layers.append(BatchNorm(output_size) if custom_batchnorm else nn.BatchNorm2d(output_size))

            layers.append(nn.MaxPool2d(2))
            layers.append(nn.ReLU(True))

        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(filter_sizes[-1] * 10 * 10, self.num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x, self.softmax(x)

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(self.num_features))
        self.beta = torch.nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x):
        mean = torch.mean(x, dim=(0,2,3))
        std = torch.std(x, dim=(0,2,3))
        x_normalized = (x - mean[None, :, None, None]) / (std[None, :, None, None] + self.eps)
        return self.gamma[None, :, None, None] * x_normalized + self.beta[None, :, None, None]

    def __call__(self, x):
        return self.forward(x)
