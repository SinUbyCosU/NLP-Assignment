import torch
import torch.nn as nn
import torch.optim as optim
import math

# setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
with open('train.csv', 'r', encoding='utf-8') as f:
    train_words = ' '.join([line.strip() for line in f if line.strip()]).split()

with open('test.csv', 'r', encoding='utf-8') as f:
    test_lines = [line.strip().split() for line in f if line.strip()]

# tokenization
vocab = {word: i + 1 for i, word in enumerate(set(train_words))}
vocab['<PAD>'] = 0  
idx2word = {i: w for w, i in vocab.items()}
vocab_size = len(vocab)

# create sequences and padding
sequences = []
for i in range(1, len(train_words)):
    sequences.append([vocab[w] for w in train_words[:i+1]])

max_len = max(len(s) for s in sequences)

X, y = [], []
for seq in sequences:
    padded = [0] * (max_len - len(seq)) + seq[:-1] 
    X.append(padded)
    y.append(seq[-1]) 

X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

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

# train
epochs = 50
print(f"Training on {device}...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_X) 
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader)
    perplexity = math.exp(avg_loss) 
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

# predict function
def predict_next_word(text):
    model.eval()
    words = text.split()
    seq = [vocab.get(w, 0) for w in words]
    
    pad_len = max(0, (max_len - 1) - len(seq))
    padded = [0] * pad_len + seq[-(max_len-1):]
    
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
        
        print(f"Test {i+1}:")
        print(f"  Input: {seed}")
        print(f"  Predicted: {predicted}")
        print(f"  Actual: {actual}")
        print(f"  Match: {predicted == actual}\n")