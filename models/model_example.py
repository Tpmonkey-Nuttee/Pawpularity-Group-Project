from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.cnn(x).squeeze()
        return self.fc_stack(x)
