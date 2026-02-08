from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.fc_h = nn.Linear(786432, 512)  # 512 x 512 x 3
        self.fc_o = nn.Linear(512, 10)
        self.fc_fuck = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        # self.fc_stack = nn.Sequential(
        #     nn.Linear(28*28, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 10)
        #     nn.Softmax(10)
        # )

    def forward(self, x):
        x = self.flatten(x)
        # print(x.shape)
        x = self.relu(self.fc_h(x))
        x = self.relu(self.fc_o(x))
        out = self.fc_fuck(x)
        # out = self.fc_stack(x)
        return self.sigmoid(out) * 100
