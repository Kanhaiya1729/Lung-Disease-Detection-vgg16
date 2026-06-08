# Lung Disease Detection Using Deep Learning

A deep learning application that classifies chest X-ray images as **Normal** or **Pneumonia** using VGG16 Transfer Learning, deployed via a Flask web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

This project addresses automated lung disease screening using computer vision. A VGG16 model, pre-trained on ImageNet and fine-tuned on chest X-ray data, is integrated into a Flask web application where users can upload an X-ray image and receive an instant classification result.

The project covers the full ML lifecycle — data preprocessing, model training, evaluation, serialization, and deployment.

---

## Dataset

| Field | Details |
|---|---|
| **Name** | Chest X-Ray Images (Pneumonia) |
| **Source** | [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) |
| **Creator** | Paul Mooney |
| **Classes** | `NORMAL`, `PNEUMONIA` |
| **Split** | Train / Validation / Test |

> Download the dataset from Kaggle and place the `chest_xray/` folder in the project root before training.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow, Keras, VGG16 |
| Image Processing | OpenCV, NumPy |
| Data Handling | Pandas, NumPy |
| Backend | Flask |
| Frontend | HTML, CSS |

---

## Project Structure

```
pythonproject/
├── app.py                    # Flask application — routing and inference
├── predict_demo.py           # Prediction pipeline — preprocessing and inference
├── vgg_transfer_learning.py  # Model definition and training
├── templates/
│   └── index.html            # Upload UI and result display
├── statics/                  # CSS and uploaded image assets
├── README.md
└── .gitignore
```

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/lung-disease-detection.git
cd lung-disease-detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install tensorflow keras flask numpy pandas opencv-python

# 4. Train the model
python vgg_transfer_learning.py

# 5. Start the application
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser.

---

## How It Works

1. User uploads a chest X-ray image through the web interface
2. Flask receives the file and passes it to the prediction pipeline
3. The image is resized to 224×224 and pixel values are normalized to `[0, 1]`
4. The trained VGG16 model runs inference on the preprocessed image
5. The predicted class (`NORMAL` or `PNEUMONIA`) is returned and displayed

---

## Why VGG16 Transfer Learning?

Training a CNN from scratch on medical images requires large amounts of labeled data and significant compute. VGG16, pre-trained on 1.2M ImageNet images, has already learned robust feature representations — edges, textures, and spatial patterns — that transfer well to chest X-ray analysis.

By freezing the convolutional base and training only the custom classification head, the model achieves strong performance with faster training and better generalization on a smaller dataset.

---

## Future Improvements

- Extend to multi-class classification (COVID-19, Tuberculosis, etc.)
- Add Grad-CAM heatmaps for model explainability
- Expose a REST API for third-party integration
- Containerize with Docker for portable deployment

---

## Author

**Your Name**  
B.Tech, Computer Science & Engineering  
[GitHub](https://github.com/Kanhaiya1729) · [LinkedIn](https://www.linkedin.com/in/kanhaiya-sahani-11487b275/) · kanhaiya.sahani2019@gmail.com

---

## License

This project is licensed under the [MIT License](LICENSE).

