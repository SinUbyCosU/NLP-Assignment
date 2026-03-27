import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Concatenate, RNN, GRUCell, SimpleRNNCell
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ==========================================
# 1. LOAD AND CLEAN THE DATASET
# ==========================================
file_path = '/kaggle/input/datasets/tanushreeyadavaghhhh/pashtoon-dataset/Data.csv'
df = pd.read_csv(file_path, on_bad_lines='skip')
df.columns = df.columns.str.strip() 
df = df.dropna(subset=['English', 'Pashto'])

input_texts = df['English'].astype(str).tolist()
target_texts = ['<start> ' + text + ' <end>' for text in df['Pashto'].astype(str).tolist()]

# ==========================================
# 2. TOKENIZATION & PADDING
# ==========================================
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
max_encoder_seq_length = max([len(txt) for txt in input_sequences])
encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
num_encoder_tokens = len(input_tokenizer.word_index) + 1

target_tokenizer = Tokenizer(filters='') 
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
max_decoder_seq_length = max([len(txt) for txt in target_sequences])
decoder_input_data = pad_sequences(target_sequences, maxlen=max_decoder_seq_length, padding='post')
num_decoder_tokens = len(target_tokenizer.word_index) + 1

decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

reverse_target_word_index = {i: word for word, i in target_tokenizer.word_index.items()}

# ==========================================
# 3. THE DYNAMIC ARCHITECTURE FUNCTION
# ==========================================
def run_translation_experiment(rnn_type='lstm', epochs=30):
    print(f"\n{'='*40}")
    print(f" INITIALIZING {rnn_type.upper()} MODEL ")
    print(f"{'='*40}")
    
    latent_dim = 256
    embedding_dim = 100

    # --- ENCODER ---
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(num_encoder_tokens, embedding_dim)(encoder_inputs)
    
    if rnn_type == 'rnn':
        # FIX: Bypassing buggy wrapper with RNN(SimpleRNNCell)
        encoder_outputs, state_h = RNN(SimpleRNNCell(latent_dim), return_state=True)(enc_emb)
        encoder_states = [state_h]
    elif rnn_type == 'gru':
        # FIX: Bypassing buggy wrapper with RNN(GRUCell)
        encoder_outputs, state_h = RNN(GRUCell(latent_dim), return_state=True)(enc_emb)
        encoder_states = [state_h]
    elif rnn_type == 'lstm':
        encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)
        encoder_states = [state_h, state_c]
    elif rnn_type == 'bilstm':
        encoder_outputs, fh, fc, bh, bc = Bidirectional(LSTM(latent_dim // 2, return_state=True))(enc_emb)
        state_h = Concatenate()([fh, bh])
        state_c = Concatenate()([fc, bc])
        encoder_states = [state_h, state_c]
    elif rnn_type == 'stacked_lstm':
        lstm1_out = LSTM(latent_dim, return_sequences=True)(enc_emb)
        encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(lstm1_out)
        encoder_states = [state_h, state_c]

    # --- DECODER ---
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(num_decoder_tokens, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)
    
    if rnn_type == 'rnn':
        decoder_layer = RNN(SimpleRNNCell(latent_dim), return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_layer(dec_emb, initial_state=encoder_states)
    elif rnn_type == 'gru':
        decoder_layer = RNN(GRUCell(latent_dim), return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_layer(dec_emb, initial_state=encoder_states)
    else:
        decoder_layer = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_layer(dec_emb, initial_state=encoder_states)

    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Compile & Train
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    print(f"Training {rnn_type.upper()}...")
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=32, epochs=epochs, validation_split=0.1, verbose=0) 

    # --- INFERENCE SETUP ---
    encoder_model = Model(encoder_inputs, encoder_states)

    if rnn_type in ['rnn', 'gru']:
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h]
        dec_emb_inf = dec_emb_layer(decoder_inputs)
        decoder_outputs_inf, state_h_inf = decoder_layer(dec_emb_inf, initial_state=decoder_states_inputs)
        decoder_states_inf = [state_h_inf]
    else:
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        dec_emb_inf = dec_emb_layer(decoder_inputs)
        decoder_outputs_inf, state_h_inf, state_c_inf = decoder_layer(dec_emb_inf, initial_state=decoder_states_inputs)
        decoder_states_inf = [state_h_inf, state_c_inf]

    decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs_inf] + decoder_states_inf)

    return encoder_model, decoder_model, rnn_type

# ==========================================
# 4. TRANSLATION AND SCORING
# ==========================================
def translate_and_score(encoder_model, decoder_model, rnn_type, test_index=3):
    english_sentence = df['English'].iloc[test_index]
    true_pashto = df['Pashto'].iloc[test_index]
    
    seq = input_tokenizer.texts_to_sequences([english_sentence])
    padded_seq = pad_sequences(seq, maxlen=max_encoder_seq_length, padding='post')
    states_value = encoder_model.predict(padded_seq, verbose=0)
    
    if not isinstance(states_value, list):
        states_value = [states_value]
        
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = ""
    
    while not stop_condition:
        predictions = decoder_model.predict([target_seq] + states_value, verbose=0)
        output_tokens = predictions[0]
        states_value = predictions[1:] 
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')
        
        if sampled_word == '<end>' or len(decoded_sentence.split()) >= max_decoder_seq_length:
            stop_condition = True
        else:
            decoded_sentence += " " + sampled_word
            
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

    predicted_pashto = decoded_sentence.strip()
    
    smoothie = SmoothingFunction().method4 
    bleu_score = sentence_bleu([true_pashto.split()], predicted_pashto.split(), smoothing_function=smoothie)
    
    print(f"English:    {english_sentence}")
    print(f"True:       {true_pashto}")
    print(f"Predicted:  {predicted_pashto}")
    print(f"BLEU Score: {bleu_score:.4f}")

# ==========================================
# 5. RUN THE EXPERIMENTS!
# ==========================================
models_to_test = ['rnn', 'gru', 'lstm', 'bilstm', 'stacked_lstm']

for model_name in models_to_test:
    enc, dec, m_type = run_translation_experiment(rnn_type=model_name, epochs=30)
    translate_and_score(enc, dec, m_type, test_index=3) 
    translate_and_score(enc, dec, m_type, test_index=10)