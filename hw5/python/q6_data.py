import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, data, data_type='train'):
        if data_type == "train":
            self.X = data['train_data']
            self.y = data['train_labels']
        else:
            self.X = data['test_data']
            self.y = data['test_labels']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

