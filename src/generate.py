import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "modelFou.keras"
TOKENIZER_PATH = "tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

max_seq_len = model.input_shape[1] + 1

total_words = len(tokenizer.word_index) + 1


def generate_text(seed_text, next_words=25, temperature=1.1):
    output = seed_text.lower()

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")

        preds = model.predict(token_list, verbose=0)[0]

        preds = np.log(preds + 1e-10) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        next_index = np.random.choice(total_words, p=preds)
        next_word = tokenizer.index_word.get(next_index, "")

        output += " " + next_word

    return output

if __name__ == "__main__":
    print("\n\n\n\n\n\n\n\n\n-------------------------------------------------------------------------")
    print(generate_text("pourquoi le ciel est bleu?\n"))
    print()
    print(generate_text("pourquoi le ciel est bleu?\n"))
    print()
    print(generate_text("comment lutter contre les big tech?\n"))
