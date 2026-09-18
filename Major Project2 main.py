import argparse
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
MAX_VOCAB_SIZE = 20000
MAX_LENGTH = 100
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_column(df, possible_names):
    normalized = {
        re.sub(r"[^a-z0-9]", "", column.lower()): column
        for column in df.columns
    }

    for name in possible_names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in normalized:
            return normalized[key]

    for column in df.columns:
        clean_column = re.sub(r"[^a-z0-9]", "", column.lower())
        for name in possible_names:
            clean_name = re.sub(r"[^a-z0-9]", "", name.lower())
            if clean_name in clean_column or clean_column in clean_name:
                return column

    return None


def clean_text(text):
    """Lowercase text and remove punctuation/numbers."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_reviews(csv_path):
    df = pd.read_csv(csv_path)

    text_column = find_column(
        df,
        [
            "Review Text",
            "Review_Text",
            "ReviewText",
            "Text",
            "Review",
            "Description",
        ],
    )

    label_column = find_column(
        df,
        [
            "Recommended IND",
            "Recommended_IND",
            "Recommended",
            "Label",
            "Sentiment",
        ],
    )

    if text_column is None or label_column is None:
        raise ValueError(
            "Could not detect review-text and label columns. "
            f"Available columns: {list(df.columns)}"
        )

    data = df[[text_column, label_column]].copy()
    data.columns = ["text", "label"]
    data["text"] = data["text"].fillna("").apply(clean_text)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna()
    data["label"] = data["label"].astype(int)

    data = data[data["text"].str.len() > 0]
    data = data[data["label"].isin([0, 1])]
    data = data.reset_index(drop=True)

    return data


def tokenize(text):
    return text.split()


def build_vocabulary(texts, max_vocab_size=MAX_VOCAB_SIZE):
    frequencies = {}

    for text in texts:
        for token in tokenize(text):
            frequencies[token] = frequencies.get(token, 0) + 1

    sorted_words = sorted(
        frequencies.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    vocabulary = {"<PAD>": 0, "<UNK>": 1}

    for word, _ in sorted_words[: max_vocab_size - 2]:
        vocabulary[word] = len(vocabulary)

    return vocabulary


def encode_text(text, vocabulary, max_length=MAX_LENGTH):
    tokens = tokenize(text)
    ids = [
        vocabulary.get(token, vocabulary["<UNK>"])
        for token in tokens[:max_length]
    ]

    if not ids:
        ids = [vocabulary["<UNK>"]]

    return torch.tensor(ids, dtype=torch.long)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocabulary):
        self.sequences = [
            encode_text(text, vocabulary)
            for text in texts
        ]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def collate_batch(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0,
    )
    return padded_sequences, torch.stack(labels)


class SequentialClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        model_type="RNN",
    ):
        super().__init__()
        self.model_type = model_type.upper()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        if self.model_type == "RNN":
            self.sequence_model = nn.RNN(
                embedding_dim,
                hidden_dim,
                batch_first=True,
            )
        elif self.model_type == "LSTM":
            self.sequence_model = nn.LSTM(
                embedding_dim,
                hidden_dim,
                batch_first=True,
            )
        elif self.model_type == "GRU":
            self.sequence_model = nn.GRU(
                embedding_dim,
                hidden_dim,
                batch_first=True,
            )
        else:
            raise ValueError("model_type must be RNN, LSTM, or GRU")

        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.sequence_model(embedded)

        if self.model_type == "LSTM":
            hidden_state = hidden[0][-1]
        else:
            hidden_state = hidden[-1]

        hidden_state = self.dropout(hidden_state)
        logits = self.output_layer(hidden_state).squeeze(1)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = (torch.sigmoid(logits) >= 0.5).int()

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    return total_loss / len(loader.dataset), accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        logits = model(sequences)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        predictions = (torch.sigmoid(logits) >= 0.5).int()

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    return (
        total_loss / len(loader.dataset),
        accuracy,
        all_labels,
        all_predictions,
    )


def train_model(model, train_loader, test_loader, device, epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy, _, _ = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    training_time = time.perf_counter() - start_time
    return pd.DataFrame(history), training_time


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def plot_history(history, model_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        label="Training Accuracy",
    )
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
    plt.savefig(
        output_dir / f"{model_name.lower()}_accuracy.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Training Loss",
    )
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
    plt.savefig(
        output_dir / f"{model_name.lower()}_loss.png",
        dpi=300,
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print("Using device:", device)

    data = load_reviews(args.csv_path)
    print("Total reviews:", len(data))
    print("Label distribution:")
    print(data["label"].value_counts())

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"].tolist(),
        data["label"].tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=data["label"],
    )

    vocabulary = build_vocabulary(train_texts)
    print("Vocabulary size:", len(vocabulary))

    train_dataset = ReviewDataset(
        train_texts,
        train_labels,
        vocabulary,
    )
    test_dataset = ReviewDataset(
        test_texts,
        test_labels,
        vocabulary,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    results = []

    for model_name in ["RNN", "LSTM", "GRU"]:
        print("\n" + "=" * 60)
        print(f"Training {model_name}")
        print("=" * 60)

        model = SequentialClassifier(
            vocab_size=len(vocabulary),
            model_type=model_name,
        ).to(device)

        history, training_time = train_model(
            model,
            train_loader,
            test_loader,
            device,
            args.epochs,
        )

        criterion = nn.BCEWithLogitsLoss()
        _, validation_accuracy, labels, predictions = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        plot_history(
            history,
            model_name,
            args.output_dir,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            model.state_dict(),
            output_dir / f"{model_name.lower()}_weights.pth",
        )

        report = classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            output_dir / f"{model_name.lower()}_classification_report.csv"
        )

        results.append({
            "Model": model_name,
            "Training_Time_Seconds": round(training_time, 2),
            "Validation_Accuracy": round(float(validation_accuracy), 4),
            "Trainable_Parameters": count_parameters(model),
        })

    comparison = pd.DataFrame(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output_dir / "rnn_lstm_gru_comparison.csv",
        index=False,
    )

    with open(output_dir / "vocabulary_size.txt", "w") as file:
        file.write(str(len(vocabulary)))

    print("\nFINAL COMPARISON")
    print(comparison.to_string(index=False))
    print("\nAll results saved in:", output_dir.resolve())


if __name__ == "__main__":
    main()
