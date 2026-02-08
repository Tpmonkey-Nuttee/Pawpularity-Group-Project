import torch
from torch.utils.data import DataLoader

from dataset import train_dataset, test_dataset
from models.model_example import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).float()

        # Compute prediction error
        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report
        train_loss += loss.item()

        if batch % 16 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    print(f"Train Error: \n\tAvg loss: {train_loss:>8f} \n")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze()
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n\tAvg loss: {test_loss*100:>8f}")


batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = NeuralNetwork().to(device)
mse = torch.nn.MSELoss()


def loss_fn(y, y_hat):
    return torch.sqrt(mse(y, y_hat))


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 2


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
