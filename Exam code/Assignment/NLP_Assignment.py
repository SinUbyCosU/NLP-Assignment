import math
import os
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

epochs = int(os.getenv("EPOCHS", "50"))

  # Load data
with open("train.csv", "r") as f:
    train_words = f.read().split()
    
with open("val.csv", "r") as f:
    val_words = f.read().split()
with open("test.csv", "r") as f:
    test_lines = [line.strip().split() for line in f if line.strip()]

#  vocabulary build
vocab = {"<pad>": 0}
for word in set(train_words):
    vocab[word] = len(vocab)

vocab_size = len(vocab)
word_to_idx = vocab
idx_to_word = {i: w for w, i in vocab.items()}

# create sequences
def make_sequences(words):
    sequences = []
    for i in range(1, len(words)):
        sequences.append(words[:i+1])
    return sequences

train_sequences = make_sequences(train_words)
val_sequences = make_sequences(val_words)

# Find max length
max_length = max(len(seq) for seq in train_sequences)

# Prepare data
def prepare_data(sequences):
    X, y = [], []
    for seq in sequences:
        if len(seq) < 2:
            continue
        # Convert to indices
        indices = [word_to_idx.get(w, 0) for w in seq]
        # Pad if needed
        while len(indices) < max_length:
            indices.insert(0, 0)
        X.append(indices[:-1])
        y.append(indices[-1])
    return torch.tensor(X), torch.tensor(y)

X_train, y_train = prepare_data(train_sequences)
X_val, y_val = prepare_data(val_sequences)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train), 
    batch_size=batch_size, 
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_val), 
    batch_size=batch_size
)

# Model
class NextWordModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

model = NextWordModel(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            val_loss += criterion(output, y_batch).item()

    if (epoch + 1) % 5 == 0:
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"epoch {epoch+1}/{epochs} - train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")

# test
def predict_next_word(text):
    model.eval()
    words = text.split()
    indices = [word_to_idx.get(w, 0) for w in words]
    
    while len(indices) < max_length - 1:
        indices.insert(0, 0)
    
    x = torch.tensor([indices]).to(device)
    with torch.no_grad():
        output = model(x)
        prediction = output.argmax(dim=1).item()
    
    return idx_to_word.get(prediction, "<unk>")

print("\ntest Results:")
for i, line in enumerate(test_lines[:5]):
    if len(line) >= 2:
        context = " ".join(line[:-1])
        predicted = predict_next_word(context)
        actual = line[-1]
        match = "Y" if predicted == actual else "N"
        print(f"{i+1}. '{context}' predicted: '{predicted}' (actual: '{actual}') {match}")