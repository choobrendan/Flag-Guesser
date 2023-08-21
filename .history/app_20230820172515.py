from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import imghdr
from matplotlib import pyplot as plt
import io

app = Flask(__name__)

# Load the pre-trained CNN model
model = tf.keras.models.load_model('static/cnn_model.h5')
model.summary()

# Define a route to render the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle image predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No image selected'})

        img = Image.open(image)
        img = img.resize((64, 64))  # Resize to match the input size of the CNN model
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize pixel values

        # Make a prediction using the CNN model
        prediction = model.predict(np.expand_dims(img_array, axis=0))

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
