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
