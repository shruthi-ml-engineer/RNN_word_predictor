import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import os

# Load and preprocess text
with open('data/lyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in text.split('.'):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        n_gram_seq = tokens[:i+1]
        input_sequences.append(n_gram_seq)

# Pad sequences
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

# Inputs and labels
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Model
model = Sequential([
    Embedding(total_words, 64, input_length=max_seq_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xs, ys, epochs=30, verbose=1)

# Save model and tokenizer
model.save('model/lstm_model.h5')
import pickle
with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
