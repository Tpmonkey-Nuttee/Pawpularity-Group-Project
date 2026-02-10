# Config file for training model

from models import model_example
from dataset import _all_dataset

import torch
from torch.utils.data import DataLoader

######## Trainer Settings ########
epoch = 4
batch_size = 8
save_after_finished_training = False
save_file_name = "./models/model_example.pth"

######## Dataset, Train/Test ratio ########
train_dataset, test_dataset = torch.utils.data.random_split(_all_dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

######## Your Model ########
model = model_example.NeuralNetwork()  # DON'T FORGET TO PUT () BEHIND THE CLASS NAME


######## Loss function ########
mse = torch.nn.MSELoss()


def rmse(y, y_hat):
    return torch.sqrt(mse(y, y_hat))


loss_fn = rmse

######## Optimizer ########
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
