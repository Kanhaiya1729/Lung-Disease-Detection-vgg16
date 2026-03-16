from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

app = Flask(__name__)

# Load model
model = load_model('vgg16_pneumonia_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file temporarily
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    prediction = model.predict(x)[0][0]
    result = 'Pneumonia' if prediction > 0.5 else 'Normal'

    # Clean up
    os.remove(filepath)

    return jsonify({'prediction': result, 'confidence': float(prediction)})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
