from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import io

app = Flask(__name__)

# Load the pre-trained CNN model
model = tf.keras.models.load_model('static/cnn_model.h5')
model.summary()

# Load class names
class_names = [...]  # Provide your list of class names here

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

        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)

        predictions = model.predict(img)
        prediction_scores = tf.nn.softmax(predictions[0])
        prediction_scores = prediction_scores.numpy()

        top_classes = np.argsort(prediction_scores)[-5:][::-1]  # Get top 5 classes

        results = []
        for class_index in top_classes:
            class_name = class_names[class_index]
            score = prediction_scores[class_index]
            results.append({'class_name': class_name, 'score': score})

        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
