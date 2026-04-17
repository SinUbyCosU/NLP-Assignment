import math
import os
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = int(os.getenv("EPOCHS", "50"))

# read files directly
with open("train.csv", "r", encoding="utf-8") as f:
    train_words = f.read().split()
with open("val.csv", "r", encoding="utf-8") as f:
    val_words = f.read().split()
with open("test.csv", "r", encoding="utf-8") as f:
    test_lines = [line.strip().split() for line in f if line.strip()]

# build vocabulary
vocab = {w: i + 1 for i, w in enumerate(set(train_words))}
vocab["<pad>"] = 0
idx2word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)

# helper to generate sequences
def get_seqs(words):
    return [[vocab.get(w, 0) for w in words[:i+1]] for i in range(1, len(words))]

train_seqs = get_seqs(train_words)
val_seqs = get_seqs(val_words)

max_len = max(len(s) for s in train_seqs)
seq_len = max_len - 1

# pad and tensorize
def pad_data(seqs):
    x_data, y_data = [], []
    for s in seqs:
        if len(s) < 2: continue
        # pre-pad with zeros and truncate to seq_len
        padded = ([0] * max(0, max_len - len(s)) + s[:-1])[-seq_len:]
        x_data.append(padded)
        y_data.append(s[-1])
    return torch.tensor(x_data, dtype=torch.long), torch.tensor(y_data, dtype=torch.long)

X_train, y_train = pad_data(train_seqs)
X_val, y_val = pad_data(val_seqs)

# dataloaders
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

# model definition
class NextWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 50, padding_idx=0)
        self.lstm = nn.LSTM(50, 100, batch_first=True, dropout=0.2, num_layers=2)
        self.fc = nn.Linear(100, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(out[:, -1, :]) # just grab the last time step

model = NextWordModel().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# training loop
for ep in range(epochs):
    model.train()
    t_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            v_loss += criterion(model(x), y).item()

    # log stats every 5 epochs
    if (ep + 1) % 5 == 0:
        avg_t = t_loss / len(train_loader)
        avg_v = v_loss / len(val_loader) if len(val_loader) else 0
        
        print(f"epoch {ep+1}/{epochs} | "
              f"train loss: {avg_t:.4f} perp: {math.exp(avg_t):.2f} | "
              f"val loss: {avg_v:.4f} perp: {math.exp(avg_v):.2f}")

# test prediction helper
def predict(text):
    model.eval()
    s = [vocab.get(w, 0) for w in text.split()]
    padded = ([0] * max(0, seq_len - len(s)) + s)[-seq_len:]
    
    with torch.no_grad():
        x = torch.tensor([padded], dtype=torch.long).to(device)
        idx = torch.argmax(model(x), dim=-1).item()
        
    return idx2word.get(idx, "<unk>")

# run tests
print("\ntest")
for i, line in enumerate(test_lines[:5]):
    if len(line) >= 2:
        seed = " ".join(line[:-1])
        pred = predict(seed)
        actual = line[-1]
        
        print(f"test {i+1}: input '{seed}' -> pred '{pred}' | actual: '{actual}' | match: {pred == actual}")