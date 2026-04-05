import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM, GRU, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============ DATA PREPARATION (Same for all models) ============
text = "deep learning is a subset of machine learning"

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index) + 1

# Create input sequences
sequences = []
words = text.split()
for i in range(1, len(words)):
    seq = tokenizer.texts_to_sequences([' '.join(words[:i+1])])[0]
    sequences.append(seq)

# Padding
max_len = max(len(s) for s in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

# Split into X and y
X = sequences[:, :-1]
y = tf.keras.utils.to_categorical(sequences[:, -1], num_classes=vocab_size)

# ============ MODEL ARCHITECTURES ============

# 1. SIMPLE RNN
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    SimpleRNN(50),
    Dense(vocab_size, activation='softmax')
])

# 2. LSTM
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    LSTM(50),
    Dense(vocab_size, activation='softmax')
])

# 3. GRU
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    GRU(50),
    Dense(vocab_size, activation='softmax')
])

# 4. STACKED LSTM (2 layers)
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    LSTM(50, return_sequences=True),  # Must return sequences for next LSTM
    LSTM(50),
    Dense(vocab_size, activation='softmax')
])

# 5. BIDIRECTIONAL LSTM
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    Bidirectional(LSTM(50)),
    Dense(vocab_size, activation='softmax')
])

# 6. LSTM WITH DROPOUT (prevents overfitting)
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    LSTM(50),
    Dropout(0.2),  # 20% dropout
    Dense(vocab_size, activation='softmax')
])

# 7. STACKED LSTM WITH DROPOUT
model = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    LSTM(50, return_sequences=True, dropout=0.2),  # dropout inside LSTM
    LSTM(50, dropout=0.2),
    Dense(vocab_size, activation='softmax')
])

# ============ TRAINING (Same for all) ============
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(X, y, epochs=100, verbose=0)

# ============ PREDICTION FUNCTION ============
def predict_next_word(model, text, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')
    pred = model.predict(seq, verbose=0)
    predicted_idx = np.argmax(pred)
    
    for word, idx in tokenizer.word_index.items():
        if idx == predicted_idx:
            return word
    return None

# Test
print(predict_next_word(model, "deep learning", tokenizer, max_len))