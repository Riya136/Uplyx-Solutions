import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = 150
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_dataset_folders(data_dir):
    """Find train and test folders for common Intel dataset layouts."""
    data_dir = Path(data_dir)

    possible_train = [
        data_dir / "seg_train" / "seg_train",
        data_dir / "seg_train",
        data_dir / "train",
    ]
    possible_test = [
        data_dir / "seg_test" / "seg_test",
        data_dir / "seg_test",
        data_dir / "test",
    ]

    train_dir = next((p for p in possible_train if p.exists()), None)
    test_dir = next((p for p in possible_test if p.exists()), None)

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            "Could not find train/test folders. Check --data_dir and dataset structure."
        )

    return train_dir, test_dir


def create_dataloaders(data_dir, batch_size=BATCH_SIZE):
    train_dir, test_dir = find_dataset_folders(data_dir)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, train_dataset.classes


class ANNClassifier(nn.Module):
    """Baseline fully connected neural network."""

    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class CNNClassifier(nn.Module):
    """CNN with convolution, max-pooling and dropout layers."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 256),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_model(model, train_loader, test_loader, device, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": test_loss,
            "validation_accuracy": test_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Validation Acc: {test_accuracy:.4f}"
        )

    training_time = time.perf_counter() - start_time
    return pd.DataFrame(history), training_time


def plot_training_history(history, model_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_accuracy"], label="Train Accuracy")
    plt.plot(
        history["epoch"],
        history["validation_accuracy"],
        label="Validation Accuracy",
    )
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation Loss",
    )
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_loss.png", dpi=300)
    plt.close()


def save_model(model, model_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"{model_name.lower()}_weights.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to Intel dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
    )
    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(test_loader.dataset)}")

    models = {
        "ANN": ANNClassifier(len(class_names)).to(device),
        "CNN": CNNClassifier(len(class_names)).to(device),
    }

    comparison = []
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        history, training_time = train_model(
            model,
            train_loader,
            test_loader,
            device,
            epochs=args.epochs,
        )

        plot_training_history(history, model_name, args.output_dir)
        save_model(model, model_name, args.output_dir)

        final_accuracy = history["validation_accuracy"].iloc[-1]
        comparison.append({
            "Model": model_name,
            "Training_Time_Seconds": round(training_time, 2),
            "Validation_Accuracy": round(float(final_accuracy), 4),
            "Trainable_Parameters": count_trainable_parameters(model),
        })

    comparison_df = pd.DataFrame(comparison)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(
        Path(args.output_dir) / "ann_vs_cnn_comparison.csv",
        index=False,
    )

    with open(Path(args.output_dir) / "class_names.json", "w") as file:
        json.dump(class_names, file, indent=4)

    print("\nFINAL COMPARISON")
    print(comparison_df.to_string(index=False))
    print(f"\nResults saved in: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
