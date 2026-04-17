import torch
import torch.nn as nn
import torch.optim as optim
import math

# setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
with open('train.csv', 'r', encoding='utf-8') as f:
    train_words = ' '.join([line.strip() for line in f if line.strip()]).split()

with open('val.csv', 'r', encoding='utf-8') as f:
    val_words = ' '.join([line.strip() for line in f if line.strip()]).split()

with open('test.csv', 'r', encoding='utf-8') as f:
    test_lines = [line.strip().split() for line in f if line.strip()]

# tokenization
vocab = {word: i + 1 for i, word in enumerate(set(train_words))}
vocab['<PAD>'] = 0  
idx2word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)

# create sequences and padding
train_sequences = []
for i in range(1, len(train_words)):
    train_sequences.append([vocab[w] for w in train_words[:i+1]])

max_len = max(len(s) for s in train_sequences)
seq_length = max_len - 1  # Input sequence length

X_train, y_train = [], []
for seq in train_sequences:
    padded = [0] * (max_len - len(seq)) + seq[:-1] 
    X_train.append(padded)
    y_train.append(seq[-1]) 

val_sequences = []
for i in range(1, len(val_words)):
    # use vocab.get(w, 0) to handle unseen validation words
    val_sequences.append([vocab.get(w, 0) for w in val_words[:i+1]])

X_val, y_val = [], []
for seq in val_sequences:
    if len(seq) < 2: continue
    pad_len = max(0, max_len - len(seq))
    padded = [0] * pad_len + seq[:-1]
    # truncate if validation sequence gets longer than train max_len
    padded = padded[-seq_length:] 
    X_val.append(padded)
    y_val.append(seq[-1])

# data loaders
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# build model
class NextWordModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=100, rnn_type='LSTM'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.2, num_layers=2)
            
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.rnn(embeds)
        return self.fc(out[:, -1, :])

model = NextWordModel(vocab_size, rnn_type='LSTM').to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# train and validate
epochs = 50
print(f"Training on {device}...")

for epoch in range(epochs):
    # training phase
    model.train()
    total_train_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_X) 
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_perp = math.exp(avg_train_loss) 
    
    # validation phase
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_val_loss += loss.item()
            
    # prevent division by zero if val dataset is extremely small
    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_perp = math.exp(avg_val_loss) if avg_val_loss > 0 else float('inf')
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Perp: {train_perp:.2f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Perp: {val_perp:.2f}")

# predict function
def predict_next_word(text):
    model.eval()
    words = text.split()
    seq = [vocab.get(w, 0) for w in words]
    
    pad_len = max(0, seq_length - len(seq))
    padded = [0] * pad_len + seq[-seq_length:]
    
    with torch.no_grad():
        x = torch.tensor([padded], dtype=torch.long).to(device)
        logits = model(x)
        predicted_idx = torch.argmax(logits, dim=-1).item()
        
    return idx2word.get(predicted_idx, "<UNKNOWN>")

# test on test data
print("\n=== Testing on test.csv ===")
for i, line in enumerate(test_lines[:5]): 
    if len(line) >= 2:
        seed = ' '.join(line[:-1])
        actual = line[-1]
        predicted = predict_next_word(seed)
        
        print(f"test {i+1}:")
        print(f"  input: {seed}")
        print(f" predicted: {predicted}")
        print(f"  Actual: {actual}")
        print(f" match: {predicted == actual}\n")