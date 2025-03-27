import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function (only applies if raw data is used)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)  # Remove special characters
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Model creation function (adjusted to output only 3 labels)
def create_model(input_length, vocab_size, embedding_dim=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.5),
        LSTM(100),
        Dense(3, activation='sigmoid')  # Only 3 output labels now
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Preprocess text data
def preprocess_text(texts, tokenizer=None, max_len=100):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_len, truncating='post')
    return data, tokenizer

# Load and prepare training data (excluding unwanted labels)
def load_data(train_csv_path='train.csv', preprocessed_csv_path='train_preprocessed.csv', max_len=100):
    if os.path.exists(preprocessed_csv_path):
        print("Loading preprocessed data...")
        df = pd.read_csv(preprocessed_csv_path)
    elif os.path.exists(train_csv_path):
        print("Loading raw data and applying text cleaning...")
        df = pd.read_csv(train_csv_path)
        df['comment_text'] = df['comment_text'].astype(str).apply(clean_text)
    else:
        raise FileNotFoundError("No training data found.")

    # Keep only the desired labels
    texts = df['comment_text'].astype(str).values
    labels = df[['toxic', 'obscene', 'insult']].values  # Keeping only 3 labels

    # Ensure labels are in binary format (0 or 1)
    labels = (labels > 0.5).astype(int)

    return texts, labels

# Train the model
def train_model(train_csv_path='train.csv', preprocessed_csv_path='train_preprocessed.csv', model_path='model.h5', tokenizer_path='tokenizer.pkl', epochs=5, batch_size=32, max_len=100):
    texts, labels = load_data(train_csv_path, preprocessed_csv_path, max_len)
    
    # Preprocess text
    data, tokenizer = preprocess_text(texts, max_len=max_len)

    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(input_length=data.shape[1], vocab_size=vocab_size)

    # Define callbacks
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # Train model
    model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint, early_stopping], verbose=1)

    # Save tokenizer
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Model training complete and saved.")

# Load trained model
def load_model_and_tokenizer(model_path='model.h5', tokenizer_path='tokenizer.pkl'):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Predict toxicity (now only outputs 3 categories)
def predict_toxicity(model, tokenizer, text, max_len=100):
    processed_text, _ = preprocess_text([text], tokenizer, max_len)
    prediction = model.predict(processed_text)[0]
    
    categories = ["Toxic", "Obscene", "Insult"]  # Only 3 categories now as the 3 other data lacked data to train
    return {cat: float(pred) for cat, pred in zip(categories, prediction)}

# Main training execution
if __name__ == "__main__":
    train_model()
