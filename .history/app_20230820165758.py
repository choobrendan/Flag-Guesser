from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import h5py
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('static/model.h5')
model.summary()

# Define a route to render the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json['data']
        data = np.array(data).reshape(1, -1)
        
        # Make a prediction using the model
        prediction = model.predict(data)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
