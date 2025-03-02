import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd
import numpy as np
import os

# Model creation function
def create_model(input_length, vocab_size, embedding_dim=100):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_shape=(input_length,)))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dense(6, activation='sigmoid'))  # 6 outputs: Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Preprocess the text data (fit tokenizer, convert to sequences)
def preprocess_text(texts, tokenizer=None, max_len=100):
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_len)
    return data, tokenizer

# Prediction function
def predict_toxicity(model, tokenizer, text, max_len=100):
    processed_text, _ = preprocess_text([text], tokenizer, max_len)
    prediction = model.predict(processed_text)
    result = {
        "Toxic": float(prediction[0][0]),
        "Severe Toxic": float(prediction[0][1]),
        "Obscene": float(prediction[0][2]),
        "Threat": float(prediction[0][3]),
        "Insult": float(prediction[0][4]),
        "Identity Hate": float(prediction[0][5]),
    }
    return result

# Save and load functions for the model
def save_model(model, tokenizer, model_path='model.h5', tokenizer_path='tokenizer.pkl'):
    model.save(model_path)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_model_and_tokenizer(model_path='model.h5', tokenizer_path='tokenizer.pkl'):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Function to train the model
def train_model(train_csv_path='train.csv', model_path='model.h5', tokenizer_path='tokenizer.pkl', epochs=5, batch_size=32, max_len=100):
    # Check if the training data file exists
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training data file not found: {train_csv_path}")

    # Load your training data
    train_data = pd.read_csv(train_csv_path)

    # Correct column names: 'comment_text' for text and the toxicity labels
    texts = train_data['comment_text'].values
    labels = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

    # Check the shape of the data and labels before any processing
    print("Data shape:", texts.shape)
    print("Labels shape before any processing:", labels.shape)

    # Ensure the number of labels per sample is 6 (for each category of toxicity)
    if labels.shape[1] != 6:
        raise ValueError(f"Expected 6 label columns, but got {labels.shape[1]}. Check your CSV file.")

    # Verify that labels are binary (0 or 1)
    if not np.all((labels >= 0) & (labels <= 1)):
        raise ValueError("Labels should be binary (0 or 1). Check your CSV file.")

    # Preprocess the text data
    data, tokenizer = preprocess_text(texts, max_len=max_len)

    # Print shape of data after preprocessing
    print("Processed data shape:", data.shape)

    # Ensure labels are in the correct shape
    print("Labels shape after processing:", labels.shape)

    # Check if the data and labels match in cardinality
    assert data.shape[0] == labels.shape[0], f"Data and labels must have the same number of samples: {data.shape[0]} != {labels.shape[0]}"

    # Create the model
    input_length = data.shape[1]
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(input_length, vocab_size)

    # Build the model and print the summary
    model.build((None, input_length))
    print(model.summary())

    # Train the model with verbose=1 for detailed output
    model.fit(data, labels, epochs=epochs, batch_size=batch_size, verbose=1)

    # Save the trained model and tokenizer
    save_model(model, tokenizer, model_path, tokenizer_path)

    print("Model training complete and saved.")

# Main execution block for training the model
if __name__ == "__main__":
    train_model(train_csv_path='train.csv', model_path='model.h5', tokenizer_path='tokenizer.pkl', epochs=5, batch_size=32)
