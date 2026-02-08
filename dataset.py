import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import torch


class ImageDataset(Dataset):
    IMAGE_SIZE = (512, 512)

    def __init__(self, df):
        self._X = df["Id"].values
        self._y = None
        if "Pawpularity" in df.keys():
            self._y = df["Pawpularity"].values
        self._transform = transforms.Resize(self.IMAGE_SIZE)
        self._convert = transforms.ConvertImageDtype(torch.float)

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        image_path = self._X[idx]
        image = read_image(f"./dataset/train/{image_path}.jpg")
        image = self._convert(self._transform(image))
        if self._y is not None:
            label = self._y[idx]
            return image, label
        return image


all_dataset = ImageDataset(
    pd.read_csv("./dataset/train.csv")
)

train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [0.8, 0.2])
