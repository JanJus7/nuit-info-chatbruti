import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


DATASET_PATH = "../dataset/dataset_chatbot_fou_20000l.txt"
MODEL_PATH = "modelFou.keras"
TOKENIZER_PATH = "tokenizer.pkl"

print("GPUs disponibles :", tf.config.list_physical_devices("GPU"))

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
total_words = len(tokenizer.word_index) + 1

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

input_sequences = []

for line in lines:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(2, len(token_list) + 1):
        seq = token_list[:i]
        input_sequences.append(seq)

max_seq_len = max(len(x) for x in input_sequences)
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre")
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]


model = Sequential(
    [
        Embedding(total_words, 256),
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        LSTM(512),
        Dropout(0.3),
        Dense(total_words, activation="softmax"),
    ]
)


model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="loss", patience=6, restore_best_weights=True)

model.fit(
    X,
    y,
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
)


model.save(MODEL_PATH)
print("Model saved to:", MODEL_PATH)
