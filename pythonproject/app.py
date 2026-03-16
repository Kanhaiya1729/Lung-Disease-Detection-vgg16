from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename
import os
import base64

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

model = load_model('vgg16_pneumonia_model.keras')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_base64(filepath):
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.route('/')
def index():
    # Initial render with no prediction and no image
    return render_template('index.html', prediction="", confidence="", image=None)

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="Error: No file uploaded", confidence="", image=None)

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="Error: No file selected", confidence="", image=None)

    if not allowed_file(file.filename):
        return render_template('index.html', prediction="Error: Invalid file type", confidence="", image=None)

    filename = secure_filename(file.filename)
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        prediction_prob = model.predict(x)[0][0]
        result = 'Pneumonia' if prediction_prob > 0.5 else 'Normal'
        confidence = round(prediction_prob * 100, 2) if prediction_prob > 0.5 else round((1 - prediction_prob) * 100, 2)

        encoded_img = encode_image_base64(filepath)

        return render_template('index.html',
                               prediction=result,
                               confidence=confidence,
                               image=encoded_img)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", confidence="", image=None)

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)
