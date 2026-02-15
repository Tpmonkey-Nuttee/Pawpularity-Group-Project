import torch
from torch import nn
import torchvision.models as models

# Nou's model
# This will cook your pc.
# It cooked colab's servers too.

# What did I try? (There are more but i didn't take note.)
# cnn -> transformer: 20.336339
# +meta as a query: 20.363566
# (pretrained) vit 2 layers: 19.969978
# +lss fn: 19.858506
# vit2l + lssfn + grad: 19.873903
# vit2 + meta: 19.867646
# vit + meta + aug + rsme: 19.628694 
# +more aug: 18.235249 (e5)

# Error: 18.235249

### Dataset Config
# import pandas as pd
# from torch.utils.data import Dataset
# from torchvision.io import read_image
# from torchvision import transforms
# import torch

# import numpy as np
# from sklearn.model_selection import train_test_split

# class ImageDataset(Dataset):
#     IMAGE_SIZE = (224, 224)

#     def __init__(self, df, is_train):
#         self._X = df["Id"].values

#         feature_cols = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
#         self._z = torch.tensor(df[feature_cols].values, dtype=torch.float32)

#         self._y = None
#         if "Pawpularity" in df.keys():
#             self._y = df["Pawpularity"].values
        
#         if is_train:
#             self._transform = transforms.Compose([
#                 transforms.Resize(self.IMAGE_SIZE),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#                 transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
#             ])
#         else:
#             # No augmentation for validation/test
#             self._transform = transforms.Resize(self.IMAGE_SIZE)
        
#         self._convert = transforms.ConvertImageDtype(torch.float)
#         self._normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], 
#             std=[0.229, 0.224, 0.225]
#         )

#     def __len__(self):
#         return len(self._X)

#     def __getitem__(self, idx):
#         image_path = self._X[idx]
#         image = read_image(f"./dataset/train/{image_path}.jpg")
#         image = self._transform(image)
#         image = self._convert(image)
#         image = self._normalize(image)  # Important for pretrained ViT

#         flags = self._z[idx]

#         if self._y is not None:
#             label = self._y[idx]
#             return (image, flags), label/100
#         return (image, flags)

# df = pd.read_csv("./dataset/train.csv")

# train_idx, test_idx = train_test_split(
#     np.arange(len(df)), 
#     test_size=0.2, 
#     random_state=42
# )

# train_dataset = ImageDataset(df.iloc[train_idx].reset_index(drop=True), is_train=True)
# test_dataset = ImageDataset(df.iloc[test_idx].reset_index(drop=True), is_train=False)

## Training config
# batch size: 32
# loss fn: RMSE
# optim: Adam
# lr: 1e-5
# weight decay: 1e-4



class NouModel(nn.Module):
    METADATA_DIM = 12

    def __init__(self):
        super().__init__()

        self.vit = models.vit_b_16(models.ViT_B_16_Weights.IMAGENET1K_V1)

        vit_features = self.vit.heads.head.in_features  # 768
        self.vit.heads.head = nn.Identity() # Remove classification head
        
        self.metadata_fc = nn.Sequential(
            nn.Linear(self.METADATA_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(vit_features + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, z):
        # Extract image features
        img_features = self.vit(x) # (batch, c, w, h) -> (batch, 768)
        
        # Process metadata
        meta_features = self.metadata_fc(z) # (batch, 12) -> (batch, 64)
        
        # Concatenate and predict
        combined = torch.cat([img_features, meta_features], dim=1)
        out = self.regression_head(combined) # (batch, 768+64) -> (batch, 1)
        
        return out