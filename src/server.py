from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

print("=== FILES IN WORKDIR ===")
print(os.listdir("."))
print("=== FILES IN SRC ===")
print(os.listdir("src"))


MODEL_PATH = "src/modelFouNewDatasetV1.keras"
TOKENIZER_PATH = "tokenizer.pkl"

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

max_seq_len = model.input_shape[1] + 1

total_words = len(tokenizer.word_index) + 1


def sample_with_penalties(
    preds,
    temperature=1.5,
    top_k=40,
    top_p=0.9,
    repetition_penalty=2.0,
    recent_tokens=None,
):
    if recent_tokens is None:
        recent_tokens = []

    preds = np.asarray(preds).astype(np.float64)

    for token in recent_tokens:
        preds[token] /= repetition_penalty

    preds = np.log(preds + 1e-10) / temperature
    preds = np.exp(preds)
    preds = preds / np.sum(preds)

    if top_k is not None:
        top_k_indices = preds.argsort()[-top_k:]
        preds_top_k = preds[top_k_indices]
        preds_top_k = preds_top_k / np.sum(preds_top_k)
    else:
        top_k_indices = np.arange(len(preds))
        preds_top_k = preds

    sorted_indices = top_k_indices[np.argsort(-preds_top_k)]
    sorted_probs = preds[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    cutoff = cumulative <= top_p
    nucleus_indices = sorted_indices[cutoff]

    nucleus_probs = preds[nucleus_indices]
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)

    if len(nucleus_indices) == 0:
        next_index = np.random.choice(top_k_indices, p=preds_top_k)
    else:
        next_index = np.random.choice(nucleus_indices, p=nucleus_probs)

    return next_index


def generate_text(seed_text, next_words=16):
    output = seed_text.lower()
    recent_tokens = []

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")

        preds = model.predict(token_list, verbose=0)[0]

        next_index = sample_with_penalties(
            preds,
            recent_tokens=recent_tokens,
        )

        next_word = tokenizer.index_word.get(next_index, "")
        output += " " + next_word

        recent_tokens.append(next_index)
        if len(recent_tokens) > 8:
            recent_tokens.pop(0)

    return output


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    response = generate_text(prompt)
    return jsonify({"response": response})


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
