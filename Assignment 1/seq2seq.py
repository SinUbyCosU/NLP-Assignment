import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ==========================================
# 1. LOAD AND CLEAN THE DATASET
# ==========================================
train_file_path = '../Exam code/train.csv'
test_file_path = '../Exam code/test.csv'
print(f"Loading training dataset from: {train_file_path}")
print(f"Loading testing dataset from: {test_file_path}")

# Load data, skipping lines that are broken by stray commas
train_df = pd.read_csv(train_file_path, on_bad_lines='skip')
test_df = pd.read_csv(test_file_path, on_bad_lines='skip')

# Strip hidden spaces from column headers to prevent KeyError
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# Drop empty rows
train_df = train_df.dropna(subset=['English', 'Pashto'])
test_df = test_df.dropna(subset=['English', 'Pashto'])

# Extract texts and add start/end tokens to the target language
input_texts = train_df['English'].astype(str).tolist()
target_texts = ['<start> ' + text + ' <end>' for text in train_df['Pashto'].astype(str).tolist()]

# ==========================================
# 2. TOKENIZATION & PADDING
# ==========================================
# English (Input)
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
max_encoder_seq_length = max([len(txt) for txt in input_sequences])
encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
num_encoder_tokens = len(input_tokenizer.word_index) + 1

# Pashto (Target)
target_tokenizer = Tokenizer(filters='') # Keep punctuation
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
max_decoder_seq_length = max([len(txt) for txt in target_sequences])
decoder_input_data = pad_sequences(target_sequences, maxlen=max_decoder_seq_length, padding='post')
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# Shift target data for Teacher Forcing
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

# ==========================================
# 3. BUILD AND TRAIN THE LSTM MODEL
# ==========================================
latent_dim = 256
embedding_dim = 100

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c] 

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compile & Train
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nStarting Training (50 Epochs)...")
model.fit(
    [encoder_input_data, decoder_input_data], 
    decoder_target_data,
    batch_size=32,
    epochs=50, 
    validation_split=0.1,
    verbose=1
)

# ==========================================
# 4. BUILD THE INFERENCE (TESTING) MODELS
# ==========================================
# Separate the Encoder
encoder_model = Model(encoder_inputs, encoder_states)

# Separate the Decoder
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb_inf = dec_emb_layer(decoder_inputs) 
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    dec_emb_inf, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf) 

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

reverse_target_word_index = {i: word for word, i in target_tokenizer.word_index.items()}

# ==========================================
# 5. TRANSLATION (WORD PREDICTION) FUNCTION
# ==========================================
def translate_sentence(input_sentence):
    seq = input_tokenizer.texts_to_sequences([input_sentence])
    
    if not seq or not seq[0]:
        return "[Error: Word not in vocabulary]"
        
    padded_seq = pad_sequences(seq, maxlen=max_encoder_seq_length, padding='post')
    states_value = encoder_model.predict(padded_seq, verbose=0)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = ""
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')
        
        if sampled_word == '<end>' or len(decoded_sentence.split()) >= max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += " " + sampled_word
            
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
        
    return decoded_sentence.strip()

# ==========================================
# 6. CALCULATE ERROR (BLEU SCORE) & TEST
# ==========================================
print("\n--- Translation Error Test (BLEU Score) ---")

# Test on a few samples from the dataset
test_indices = [idx for idx in [0, 2, 4] if idx < len(test_df)]
smoothie = SmoothingFunction().method4 

for idx in test_indices:
    english_sentence = test_df['English'].iloc[idx]
    true_pashto = test_df['Pashto'].iloc[idx]
    
    predicted_pashto = translate_sentence(english_sentence)
    
    # Calculate BLEU Score
    reference = [true_pashto.split()] 
    candidate = predicted_pashto.split() 
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    
    print(f"English:    {english_sentence}")
    print(f"True:       {true_pashto}")
    print(f"Predicted:  {predicted_pashto}")
    print(f"BLEU Score: {bleu_score:.4f} \n")