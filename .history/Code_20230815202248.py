from flask import Flask, render_template, request
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os

try:
	import shutil
	shutil.rmtree('uploaded / image')
	% cd uploaded % mkdir image % cd ..
	print()
except:
	pass

model = tf.keras.models.load_model('model')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded / image'

@app.route('/')
def upload_f():
	return render_template('upload.html')

def finds():
    test_datagen = ImageDataGenerator(rescale=1./255)
    vals = ["Bosnia and Herzegovina"
" Tuvalu"
"Switzerland"
"Ecuador"
"Kenya"
"Haiti"
"Myanmar"
"Lesotho"
"Guinea Bisseau"
" Uganda"
"Hungary"
" Guinea"
"Guatemala"
"Nauru"
" Maldives"
" Monaco"
"Namibia"
" Solomon Islands"
"Benin"
" Kiribati"
" Greece"
" Eswatini"
" Malaysia"
" Kuwait"
"New Zealand"
"Ireland"
"Sierra Leone"
" Laos"
"Belarus"
"Singapore"
"Italy"
"South Africa"
" Angola"
" Gambia"
" Saint Kitts and Nevis"
" Mongolia"
" Madagascar"
" Poland"
"USA"
" Dominica"
"Andorra"
" Mauritania"
" Slovakia"
"Somalia"
" Paraguay"
" Sweden"
" Malawi"
"Finland"
" Mali"
" Norway"
"Senegal"
" Serbia"
" Antigua and Barbuda"
" Brunei"
" Togo"
"Chile"
" Rwanda"
" Iran"
"Croatia"
"Ivory Coast"
"Bahrain"
"South Korea"
" North Macedonia"
"Japan"
"Saint Vincent and the Grenadines"
"Romania"
" Cape Verde"
"Luxemborg"
"Afghanistan"
"Uruguay"
"El Salvador"
"Armenia"
" Chad"
" Uzbekistan"
"Lithuania"
" Equatorial Guinea"
"Iceland"
"Venezuela"
"Modolva"
"Morocco"
" Thailand"
" Colombia"
"Timor Leste"
" Zambia"
"South Sudan"
"United Kingdom"
" Mozambique"
" St Kittis and Nevis"
" Peru"
"Grenada"
" Panama"
"Czech Republic"
"CAR"
" Suriname"
"Austria"
"Belgium"
"Liechenstein"
" Israel"
"Malta"
"Marshall Islands"
"Turkmenistan"
" Latvia"
" Tanzania"
"Liberia"
" Botswana"
"Dominican Republic"
" Brazil"
" Tajikistan"
"Bahamas"
"Doctor Congo"
" Jordan"
"Libya"
"Nigeria"
"Syria"
"Nepal"
" Guyana"
" Portugal"
"North Korea"
"Burkina Faso"
"Tunisia"
" Cyprus"
"Philippines"
"Egypt"
" Ethiopia"
"Lebanon"
" Djibouti"
"Comoros"
" Trinidad and Tobago"
" Seychelles"
"Tonga"
"Vietnam"
" Montenegro"
" Fiji"
"Sri Lanka"
"Burundi"
"Qatar"
"Spain"
" Kyrgyzstan"
" Russia"
" San Marino"
"Eritrea"
" Barbados"
" Zimbabwe"
"Samoa"
"India"
"UAE"
" Mexico"
" Bangladesh"
"Georgia"
" Iraq"
"Palau"
"Algeria"
" Canada"
" France"
" Cameroon"
"Congo"
" Costa Rica"
"Argentina"
"Papua New Guinea"
"Albania"
" Bhutan"
"Estonia"
" Slovenia"
" Belize"
"Vanuatu"
"Australia"
"Jamaica"
"Niger"
" Pakistan"
" Micronesia"
"Indonesia"
" Cuba"
"Netherlands"
"Mauritius"
" Kazakhstan"
" Azerbaijan"
"Ghana"
"China"
" Oman"
" Sao Tome and Principe"
"Yemen"
"Germany"
" Turkey"
" Bulgaria"
"Gabon"
"Honduras"
"Cambodia"
"Saint Lucia"
"Saudi Arabia"
"Ukraine"
"Sudan"
"Bolivia"]
    test_dir = 'uploaded'
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=1
    )

    pred = model.predict_generator(test_generator)
    top_classes = np.argsort(-pred[0])[:5]  # Get the indices of top 5 predicted classes

    top_categories = [vals[idx] for idx in top_classes]  # Convert indices to class labels
    confidence_rates = [pred[0][idx] for idx in top_classes]  # Get confidence rates for top classes

    return top_categories, confidence_rates

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        top_categories, confidence_rates = finds()
        return render_template('pred.html', categories_confidences=zip(top_categories, confidence_rates))


if __name__ == '__main__':
	app.run()
