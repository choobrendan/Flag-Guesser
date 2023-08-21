from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import io

app = Flask(__name__)

# Load the pre-trained CNN model
model = tf.keras.models.load_model('static/model.h5')
model.summary()

# Load class names
class_names = ['Afghanistan',
 'Albania',
 'Algeria',
 'Andorra',
 'Angola',
 'Antigua & Barbuda',
 'Argentina',
 'Armenia',
 'Australia',
 'Austria',
 'Azerbaijan',
 'Bahamas',
 'Bahrain',
 'Bangladesh',
 'Barbados',
 'Belarus',
 'Belgium',
 'Belize',
 'Benin',
 'Bhutan',
 'Bolivia',
 'Bosnia & Herzegovina',
 'Botswana',
 'Brazil',
 'Brunei',
 'Bulgaria',
 'Burkina Faso',
 'Burundi',
 'CAR',
 'Cambodia',
 'Cameroon',
 'Canada',
 'Cape Verde',
 'Chad',
 'Chile',
 'China',
 'Colombia',
 'Comoros',
 'Congo',
 'Costa Rica',
 'Croatia',
 'Cuba',
 'Cyprus',
 'Czech Republic',
 'Denmark',
 'Djibouti',
 'Doctor Congo',
 'Dominica',
 'Dominican Republic',
 'Ecuador',
 'Egypt',
 'El Salvador',
 'Equatorial Guinea',
 'Eritrea',
 'Estonia',
 'Eswatini',
 'Ethiopia',
 'Fiji',
 'Finland',
 'France',
 'Gabon',
 'Gambia',
 'Georgia',
 'Germany',
 'Ghana',
 'Greece',
 'Grenada',
 'Guatemala',
 'Guinea',
 'Guinea Bisseau',
 'Guyana',
 'Haiti',
 'Honduras',
 'Hungary',
 'Iceland',
 'India',
 'Indonesia',
 'Iran',
 'Iraq',
 'Ireland',
 'Israel',
 'Italy',
 'Ivory Coast',
 'Jamaica',
 'Japan',
 'Jordan',
 'Kazakhstan',
 'Kenya',
 'Kiribati',
 'Kuwait',
 'Kyrgyzstan',
 'Laos',
 'Latvia',
 'Lebanon',
 'Lesotho',
 'Liberia',
 'Libya',
 'Liechenstein',
 'Lithuania',
 'Luxemborg',
 'Madagascar',
 'Malawi',
 'Malaysia',
 'Maldives',
 'Mali',
 'Malta',
 'Marshall Islands',
 'Mauritius',
 'Maurituana',
 'Mexico',
 'Micronesia',
 'Modolva',
 'Monaco',
 'Mongolia',
 'Montenegro',
 'Morocco',
 'Mozambique',
 'Myanmar',
 'Namibia',
 'Nauru',
 'Nepal',
 'Netherlands',
 'New Zealand',
 'Niger',
 'Nigeria',
 'North Korea',
 'North Macedonia',
 'Norway',
 'Oman',
 'Pakistan',
 'Palau',
 'Panama',
 'Papua Nue Guinea',
 'Paraguay',
 'Peru',
 'Philippines',
 'Poland',
 'Portugal',
 'Qatar',
 'Romania',
 'Russia',
 'Rwanda',
 'Saint Kitts and Nevis',
 'Saint Lucia',
 'Saint Vincent and the Grenadines',
 'Samoa',
 'San Marino',
 'Sao Tome and Principe',
 'Saudi Arabia',
 'Senegal',
 'Serbia',
 'Seychelles',
 'Sierra Leone',
 'Singapore',
 'Slovakia',
 'Slovenia',
 'Solomon Islands',
 'Somalia',
 'South Africa',
 'South Korea',
 'South Sudan',
 'Spain',
 'Sri Lanka',
 'St Kittis and Nevis',
 'Sudan',
 'Suriname',
 'Sweden',
 'Switzerland',
 'Syria',
 'Tajikistan',
 'Tanzania',
 'Thailand',
 'Timor Leste',
 'Togo',
 'Tonga',
 'Trinidad and Tobago',
 'Tunisia',
 'Turkey',
 'Turkmenistan',
 'Tuvalu',
 'UAE',
 'USA',
 'Uganda',
 'Ukraine',
 'United Kingdom',
 'Uruguay',
 'Uzbekistan',
 'Vanuatu',
 'Venezuela',
 'Vietnam',
 'Yemen',
 'Zambia',
 'Zimbabwe']

# Define a route to render the index.html page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # try:
    #     print("aaaaaaaaaaaaa")
    #     if 'image' not in request.files:
    #         return jsonify({'error': 'No image uploaded'})

        image = request.files['image']
        # if image.filename == '':
        #     return jsonify({'error': 'No image selected'})

        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = cv2.resize(img, (128, 128))
        img = tf.expand_dims(img, axis=0)

        predictions = model.predict(img)
        prediction_scores = tf.nn.softmax(predictions[0])
        prediction_scores = prediction_scores.numpy()
        top_classes = np.argsort(prediction_scores)[-5:][::-1]  # Get top 5 classes
        
        result_strings = []
        for result in top_classes:
            class_name = result['class_name']
            score = result['score']
            result_string = f"Class: {class_name}, Score: {score}"
            result_strings.append(result_string)

        return jsonify(results=result_strings)
        results = []
        for class_index in top_classes:
            class_name = class_names[class_index]
            score = prediction_scores[class_index]
            result_string=f"Class: {class_name}, Score: {score}"
            results.append(result_string)
            # Print the class name and score to the terminal
        return jsonify(fr)
    # except Exception as e:
    #     return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
