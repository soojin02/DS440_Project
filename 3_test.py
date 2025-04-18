# test.py
import pickle
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import your functions from model.py
from model import clean_text, preprocess_text, load_data

# Load trained model and tokenizer
model = tf.keras.models.load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load and prepare validation data
texts, labels = load_data('train_preprocessed.csv')
data, _ = preprocess_text(texts, tokenizer=tokenizer)
_, x_val, _, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Evaluate model
loss, accuracy = model.evaluate(x_val, y_val, verbose=1)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# Optional: More detailed metrics
y_pred = model.predict(x_val) > 0.5
print(classification_report(y_val, y_pred, target_names=["Toxic", "Obscene", "Insult"]))
