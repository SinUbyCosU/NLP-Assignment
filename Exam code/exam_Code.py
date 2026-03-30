#NLP exam
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM, GRU, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text
text = "deep learning is a subset of machine learning which is a subset of artificial intelligence"

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index) + 1

# Create sequences
sequences = []
words = text.split()
for i in range(1, len(words)):
    seq = tokenizer.texts_to_sequences([' '.join(words[:i+1])])[0]
    sequences.append(seq)

# Pad and split X, y
max_len = max(len(s) for s in sequences)
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
X = sequences[:, :-1]
y = tf.keras.utils.to_categorical(sequences[:, -1], num_classes=vocab_size)

#SIMPLE RNN

model_rnn = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    SimpleRNN(50, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])
model_rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_rnn.fit(X, y, epochs=100, verbose=0)

#LSTM

model_lstm = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    LSTM(50),
    Dense(vocab_size, activation='softmax')
])
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(X, y, epochs=100, verbose=0)

#GRU

model_gru = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    GRU(50),
    Dense(vocab_size, activation='softmax')
])
model_gru.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_gru.fit(X, y, epochs=100, verbose=0)

#stacked LSTM

model_stacked = Sequential([
    Embedding(vocab_size, 10, input_length=max_len-1),
    LSTM(50, return_sequences=True),  # return_sequences=True for stacking
    LSTM(50),
    Dense(vocab_size, activation='softmax')
])
model_stacked.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_stacked.fit(X, y, epochs=100, verbose=0)

#Bidirectional LSTM
