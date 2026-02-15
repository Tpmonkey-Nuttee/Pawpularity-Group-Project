import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    IMAGE_SIZE = (512, 512)

    def __init__(self, df, is_train=True):
        self._X = df["Id"].values
        self._y = None
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
        
        if is_train:
            self._transform = transforms.Compose([
                transforms.Resize(self.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
        else:
            self._transform = transforms.Resize(self.IMAGE_SIZE)
        
        self._convert = transforms.ConvertImageDtype(torch.float)
        
        # Magic numbers! :)
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(f"./dataset/train/{image_path}.jpg")
        image = self._transform(image)
        image = self._convert(image)
        
        # This may or may not help you.
        # image = self._normalize(image) 
        
        if self._y is not None:
            label = self._y[idx]
            return image, label/100
        return image


df = pd.read_csv("./dataset/train.csv")

train_idx, test_idx = train_test_split(
    np.arange(len(df)), 
    test_size=0.2, 
    random_state=42
)

train_dataset = ImageDataset(df.iloc[train_idx].reset_index(drop=True), is_train=True)
test_dataset = ImageDataset(df.iloc[test_idx].reset_index(drop=True), is_train=False)
