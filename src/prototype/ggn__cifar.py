import os
from datetime import datetime
import socket
import logging
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

SYSTEM_ID = socket.gethostname()
DATETIME_ID = datetime.now().strftime("%b%d_%H-%M-%S")

logger = logging.getLogger("simple_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def load_cifar10(batch_size=32):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data/raw", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data/raw", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    writer,
    device,
    epochs=5,
    patience=3,
):
    best_val_loss = float("inf")
    patience_count = 0
    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(train_loader)
            writer.add_scalar("training_loss", avg_epoch_loss, epoch)

            val_loss, val_accuracy = validate_model(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                writer=writer,
                global_step=epoch * len(train_loader),
            )

            writer.add_scalar("validation_loss", val_loss, epoch)
            writer.add_scalar("validation_accuracy", val_accuracy, epoch)
            
            logger.info(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {avg_epoch_loss:.3f} | Validation Loss: {val_loss:.3f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                save_model(model)
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}...")
                    break
    except KeyboardInterrupt:
        logger.info(f"Training interrupted at epoch {epoch+1}...")

def validate_model(model, val_loader, criterion, device, writer, global_step):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    return val_loss, accuracy


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Accuracy on the test set: {accuracy:.2f}%")


def save_model(model, path="models"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(model.state_dict(), os.path.join(path, f"{DATETIME_ID}_{SYSTEM_ID}.pth"))


def main(epochs=10, lr=0.001):
    train_loader, test_loader = load_cifar10(batch_size=32)

    train_size = int(0.8 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    writer = SummaryWriter()

    logger.info(f"Starting model training on `{torch.cuda.get_device_name(0)}`...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        writer=writer,
        device=device,
        epochs=epochs,
    )
    logger.info("Training completed. Evaluating model on test set...")
    
    test_model(model, test_loader, device)
    save_model(model)
    
    writer.close()
    logger.info("TensorBoard writer closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--epochs", default=10, type=int, help="epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    args = vars(parser.parse_args())
    
    main(**args)
