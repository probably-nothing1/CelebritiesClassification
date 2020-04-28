import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize

from constants import TRAIN_IMAGE_MEAN, TRAIN_IMAGE_STD, TEST_IMAGE_MEAN, TEST_IMAGE_STD


def get_train_datloader(data_dir, batch_size):
  train_transform = Compose([ToTensor(), Normalize(TRAIN_IMAGE_MEAN, TRAIN_IMAGE_STD)])
  train_dataset = ImageFolder(os.path.join(data_dir, 'train/'), transform=train_transform)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def get_test_datloader(data_dir, batch_size):
  test_transform = Compose([ToTensor(), Normalize(TEST_IMAGE_MEAN, TEST_IMAGE_STD)])
  test_dataset = ImageFolder(os.path.join(data_dir, 'test/'), transform=test_transform)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
