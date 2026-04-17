import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = int(os.getenv("EPOCHS", "50"))
base_dir = Path(__file__).resolve().parent


def resolve_data_file(file_name):
    candidates = [base_dir / file_name, base_dir.parent / file_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_words(file_path):
    with open(file_path, "r", encoding="utf-8") as file_handle:
        text = " ".join(line.strip() for line in file_handle if line.strip())
    return text.split()


def read_test_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as file_handle:
        return [line.strip().split() for line in file_handle if line.strip()]


def build_vocab(words):
    vocab = {word: index + 1 for index, word in enumerate(set(words))}
    vocab["<PAD>"] = 0
    return vocab, {index: word for word, index in vocab.items()}


def build_sequences(words, vocab_map, use_default_zero=False):
    sequences = []
    for end_index in range(1, len(words)):
        if use_default_zero:
            sequence = [vocab_map.get(word, 0) for word in words[: end_index + 1]]
        else:
            sequence = [vocab_map[word] for word in words[: end_index + 1]]
        sequences.append(sequence)
    return sequences


def pad_train_sequences(sequences):
    max_len = max(len(sequence) for sequence in sequences)
    input_len = max_len - 1

    features, targets = [], []
    for sequence in sequences:
        padded = [0] * (max_len - len(sequence)) + sequence[:-1]
        features.append(padded)
        targets.append(sequence[-1])

    return features, targets, max_len, input_len


def pad_validation_sequences(sequences, max_len, input_len):
    features, targets = [], []

    for sequence in sequences:
        if len(sequence) < 2:
            continue

        padded = [0] * max(0, max_len - len(sequence)) + sequence[:-1]
        features.append(padded[-input_len:])
        targets.append(sequence[-1])

    return features, targets


train_words = read_words(resolve_data_file("train.csv"))
val_words = read_words(resolve_data_file("val.csv"))
test_lines = read_test_lines(resolve_data_file("test.csv"))

vocab, idx2word = build_vocab(train_words)
vocab_size = len(vocab)

train_sequences = build_sequences(train_words, vocab)
X_train, y_train, max_len, seq_length = pad_train_sequences(train_sequences)

val_sequences = build_sequences(val_words, vocab, use_default_zero=True)
X_val, y_val = pad_validation_sequences(val_sequences, max_len, seq_length)

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train, dtype=torch.long),
    torch.tensor(y_train, dtype=torch.long),
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_val, dtype=torch.long),
    torch.tensor(y_val, dtype=torch.long),
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class NextWordModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=100, rnn_type="LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if rnn_type == "RNN":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                batch_first=True,
                dropout=0.2,
                num_layers=2,
            )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs, _ = self.rnn(embeddings)
        return self.fc(outputs[:, -1, :])


model = NextWordModel(vocab_size, rnn_type="LSTM").to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.005)


for epoch in range(epochs):
    model.train()
    train_loss_total = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()

    avg_train_loss = train_loss_total / len(train_loader)
    train_perplexity = math.exp(avg_train_loss)

    model.eval()
    val_loss_total = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            val_loss_total += loss.item()

    avg_val_loss = val_loss_total / len(val_loader) if len(val_loader) > 0 else 0
    val_perplexity = math.exp(avg_val_loss) if avg_val_loss > 0 else float("inf")

    if (epoch + 1) % 5 == 0:
        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f}, Train Perp: {train_perplexity:.2f} | "
            f"val Loss: {avg_val_loss:.4f}, val Perp: {val_perplexity:.2f}"
        )


def predict_next_word(text):
    model.eval()
    words = text.split()
    sequence = [vocab.get(word, 0) for word in words]

    padded = [0] * max(0, seq_length - len(sequence)) + sequence[-seq_length:]

    with torch.no_grad():
        inputs = torch.tensor([padded], dtype=torch.long).to(device)
        logits = model(inputs)
        predicted_idx = torch.argmax(logits, dim=-1).item()

    return idx2word.get(predicted_idx, "<UNKNOWN>")


for index, line in enumerate(test_lines[:5]):
    if len(line) >= 2:
        seed = " ".join(line[:-1])
        actual = line[-1]
        predicted = predict_next_word(seed)

        print(f"test {index + 1}:")
        print(f"  input: {seed}")
        print(f" predicted: {predicted}")
        print(f"  Actual: {actual}")
        print(f" match: {predicted == actual}\n")