from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('static/model.h5')
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
            return jsonify({'error': 'No file part'})

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'})

        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize((12, 224))  # Resize to match the model's input size
        image_array = np.array(image) / 255.0  # Normalize pixel values

        # Make a prediction using the model
        prediction = model.predict(np.expand_dims(image_array, axis=0))

        # Assuming the model output is an index representing the class
        predicted_class = np.argmax(prediction)

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
