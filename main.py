import torch
import config

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


batch_size = config.batch_size
train_dataloader = config.train_dataloader
test_dataloader = config.test_dataloader

model = config.model().to(device)
loss_fn = config.loss_fn
optimizer = config.optimizer

for t in range(config.epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

if config.save_after_finished_training:
    torch.save(model.state_dict(), config.save_file_name)
