import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM, GRU, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============ LOAD DATA ============
with open('train.csv', 'r', encoding='utf-8') as f:
    train_text = ' '.join([line.strip() for line in f if line.strip()])

with open('test.csv', 'r', encoding='utf-8') as f:
    test_lines = [line.strip() for line in f if line.strip()]

# ============ TOKENIZATION ============
tokenizer = Tokenizer()
tokenizer.fit_on_texts([train_text])
vocab_size = len(tokenizer.word_index) + 1

# ============ CREATE SEQUENCES ============
sequences = []
words = train_text.split()
for i in range(1, len(words)):
    seq = tokenizer.texts_to_sequences([' '.join(words[:i+1])])[0]
    sequences.append(seq)

# ============ PADDING ============
max_len = max(len(s) for s in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

# ============ SPLIT X and y ============
X = sequences[:, :-1]
y = tf.keras.utils.to_categorical(sequences[:, -1], num_classes=vocab_size)

# ============ BUILD MODEL ============
# CHOOSE ONE: Comment out others, uncomment the one you want

# Option 1: Simple RNN
model = Sequential([
    Embedding(vocab_size, 50, input_length=max_len-1),
    SimpleRNN(100),
    Dense(vocab_size, activation='softmax')
])

# Option 2: LSTM
 model = Sequential([
     Embedding(vocab_size, 50, input_length=max_len-1),
     LSTM(100),
     Dense(vocab_size, activation='softmax')
 ])

# Option 3: GRU
 model = Sequential([
     Embedding(vocab_size, 50, input_length=max_len-1),
     GRU(100),
     Dense(vocab_size, activation='softmax')
 ])

# Option 4: Stacked LSTM
 model = Sequential([
     Embedding(vocab_size, 50, input_length=max_len-1),
     LSTM(100, return_sequences=True),
     LSTM(100),
     Dense(vocab_size, activation='softmax')
 ])

# Option 5: Bidirectional LSTM
# model = Sequential([
#     Embedding(vocab_size, 50, input_length=max_len-1),
#     Bidirectional(LSTM(100)),
#     Dense(vocab_size, activation='softmax')
# ])

# Option 6: LSTM with Dropout
 model = Sequential([
     Embedding(vocab_size, 50, input_length=max_len-1),
     LSTM(100, dropout=0.2, recurrent_dropout=0.2),
     Dropout(0.2),
     Dense(vocab_size, activation='softmax')
 ])

# ============ COMPILE ============
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ============ TRAIN ============
history = model.fit(X, y, epochs=100, verbose=1)

# ============ PREDICT FUNCTION ============
def predict_next_word(text):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')
    pred = model.predict(seq, verbose=0)
    predicted_idx = np.argmax(pred)
    
    for word, idx in tokenizer.word_index.items():
        if idx == predicted_idx:
            return word
    return None

# ============ TEST ON TEST DATA ============
print("\n=== Testing on test.csv ===")
for i, line in enumerate(test_lines[:5]):  # First 5 test cases
    words = line.split()
    if len(words) >= 2:
        seed = ' '.join(words[:-1])  # All words except last
        predicted = predict_next_word(seed)
        actual = words[-1]
        print(f"Test {i+1}:")
        print(f"  Input: {seed}")
        print(f"  Predicted: {predicted}")
        print(f"  Actual: {actual}")
        print(f"  Match: {predicted == actual}\n")