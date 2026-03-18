from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from flask_cors import CORS
from keras.layers import TFSMLayer
from keras.models import Sequential
import os

# Initialize Flask
app = Flask(__name__, static_folder="static")
CORS(app)

# Lazy-loaded model (IMPORTANT for Render)
model = None


def load_model():
    global model
    if model is None:
        print("Loading ML model...")
        model = Sequential(
            [TFSMLayer("saved_model/LSTMAttentionXSS", call_endpoint="serving_default")]
        )
        print("Model loaded.")


# Load tokenizer & label encoder
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("label_encoder.pkl", "rb") as handle:
    label_encoder = pickle.load(handle)

max_len = 800

# ======================
# ROUTES
# ======================


@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return "ok"


@app.route("/predict", methods=["POST"])
def predict():
    load_model()

    data = request.get_json(silent=True) or {}
    if "text" not in data:
        return jsonify({"error": "No text field provided"}), 400

    input_text = data["text"]
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequences, maxlen=max_len)

    prediction = model(padded_sequence)
    prediction = list(prediction.values())[0].numpy()

    malicious_probability = float(prediction.flatten()[0])
    predicted_label = (prediction > 0.5).astype(int)

    predicted_class_label = label_encoder.inverse_transform(predicted_label.flatten())[
        0
    ]

    confidence = (
        malicious_probability
        if malicious_probability >= 0.5
        else 1.0 - malicious_probability
    )

    return jsonify(
        {
            "value": predicted_class_label,
            "probability": malicious_probability,
            "confidence": confidence,
        }
    )


# ======================
# LOCAL DEV ONLY
# ======================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🌐 Server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
