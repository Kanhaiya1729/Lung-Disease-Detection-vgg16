from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Load saved model
model = load_model('vgg16_pneumonia_model.keras')  # Use the .keras file you saved

def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)

    # Output
    if prediction[0][0] > 0.5:
        print("Prediction: Pneumonia detected")
    else:
        print("Prediction: Normal")

# Test the function on a sample image
if __name__ == "__main__":
    image_path = "dataset/chest_xray/test"
  # Update with an actual image file path
    predict_image(image_path)
