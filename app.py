import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load model and tokenizer
model = tf.keras.models.load_model('model/lstm_model.h5')
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = model.input_shape[1]

def predict_next_word(seed_text):
    tokens = tokenizer.texts_to_sequences([seed_text])[0]
    tokens = pad_sequences([tokens], maxlen=max_len, padding='pre')
    predicted = model.predict(tokens, verbose=0)
    predicted_word = ''
    for word, index in tokenizer.word_index.items():
        if index == np.argmax(predicted):
            predicted_word = word
            break
    return predicted_word

st.title("RNN Next Word Predictor")
seed = st.text_input("Enter a sentence:")

if seed:
    next_word = predict_next_word(seed)
    st.write(f"**Next word prediction:** `{next_word}`")
