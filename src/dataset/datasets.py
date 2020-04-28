import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip, RandomResizedCrop

from .constants import TRAIN_IMAGE_MEAN, TRAIN_IMAGE_STD, TEST_IMAGE_MEAN, TEST_IMAGE_STD


def get_train_dataloader(data_dir, batch_size, augmentation=False):
  transform_list = [ToTensor(), Normalize(TRAIN_IMAGE_MEAN, TRAIN_IMAGE_STD)]
  augmentation_list = []
  if augmentation:
    augmentation_list.append(ColorJitter(brightness=0.4, hue=0.0, saturation=0.7, contrast=0.2))
    augmentation_list.append(RandomHorizontalFlip())
    augmentation_list.append(RandomResizedCrop(size=250, scale=(0.5, 1.0), ratio=(0.9, 1.1)))

  train_transform = Compose(augmentation_list + transform_list)
  train_dataset = ImageFolder(data_dir, transform=train_transform)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
  return train_dataloader

def get_test_dataloader(data_dir, batch_size):
  test_transform = Compose([ToTensor(), Normalize(TEST_IMAGE_MEAN, TEST_IMAGE_STD)])
  test_dataset = ImageFolder(data_dir, transform=test_transform)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
  return test_dataloader
