from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from flask_cors import CORS  # Import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model('my_model.h5')

# Load the tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

max_len = 800  # Ensure this is consistent with your training

@app.route('/')
def index():
    return "Halo, selamat datang dialfamart"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text field provided'}), 400

    # Preprocess the input text
    input_text = data['text']
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequences, maxlen=max_len)

    # Make prediction
    prediction = model.predict(padded_sequence)
    predicted_label = (prediction > 0.5).astype(int)

    # Decode the label
    predicted_class_label = label_encoder.inverse_transform(predicted_label.flatten())[0]

    # Return the prediction
    return jsonify({'value': predicted_class_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
