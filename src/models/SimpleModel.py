import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, num_classes=28, filter_sizes=[32, 64, 64, 128], use_bn=False, custom_batchnorm=False):
        super().__init__()
        self.num_classes = num_classes
        # modules_list = []
        # for filter_size in filter_sizes
        self.conv1 = nn.Conv2d(3, 32, 7)
        self.bn1 = BatchNorm(32) if custom_batchnorm else nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 7)
        self.bn2 = BatchNorm(64) if custom_batchnorm else nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 7)
        self.bn3 = BatchNorm(64) if custom_batchnorm else nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 7)
        self.bn4 = BatchNorm(128) if custom_batchnorm else nn.BatchNorm2d(128)
        self.linear = nn.Linear(128 * 10 * 10, self.num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
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
