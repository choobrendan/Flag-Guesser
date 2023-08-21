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

        new_image = cv2.imread(new_image_path)
        new_image = cv2.convertScaleAbs(new_image)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        new_image = tf.image.convert_image_dtype(new_image, dtype=tf.uint8)
        resize= tf.image.resize(new_image,(128,128))
        resize = tf.expand_dims(resize, 0)
        predictions=model.predict(resize)
        def find_largest_indices(arr, n):
            if n > len(arr):
                n = len(arr)
            indices = np.argsort(arr)[-n:][::-1]
            return indices
        score = tf.nn.softmax(predictions[0])
        score=score.numpy()
        # Find the indices of the 5 largest elements
        largest_indices = find_largest_indices(predictions[0], 10)
        print(largest_indices)
        for i in largest_indices:
            print(class_names[i])
            print(score[i])
        return jsonify({'prediction': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
