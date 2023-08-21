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
    vals = [Bosnia and Herzegovina"       1.00      0.92      0.96        12
                          Tuvalu"       1.00      0.96      0.98        26
                     Switzerland"       0.92      0.96      0.94        24
                         Ecuador"       0.96      1.00      0.98        25
                           Kenya"       1.00      1.00      1.00        17
                           Haiti"       1.00      0.94      0.97        33
                         Myanmar"       1.00      0.78      0.88        23
                         Lesotho"       0.93      1.00      0.96        25
                  Guinea Bisseau"       1.00      0.72      0.84        18
                          Uganda"       0.83      0.95      0.88        20
                         Hungary"       1.00      0.91      0.95        32
                          Guinea"       1.00      1.00      1.00        33
                       Guatemala"       0.86      0.91      0.88        33
                           Nauru"       0.86      1.00      0.93        19
                        Maldives"       0.94      1.00      0.97        17
                          Monaco"       0.96      1.00      0.98        24
                         Namibia"       1.00      1.00      1.00        22
                 Solomon Islands"       1.00      0.95      0.97        19
                           Benin"       0.96      0.96      0.96        23
                        Kiribati"       0.95      0.95      0.95        21
                          Greece"       0.97      0.91      0.94        34
                        Eswatini"       0.94      0.94      0.94        16
                        Malaysia"       0.90      0.84      0.87        43
                          Kuwait"       0.94      1.00      0.97        16
                     New Zealand"       1.00      1.00      1.00        21
                         Ireland"       0.91      1.00      0.95        21
                    Sierra Leone"       1.00      0.93      0.97        30
                            Laos"       1.00      1.00      1.00        35
                         Belarus"       1.00      1.00      1.00        27
                       Singapore"       1.00      0.96      0.98        24
                           Italy"       1.00      0.96      0.98        26
                    South Africa"       0.90      1.00      0.95        38
                          Angola"       0.96      0.92      0.94        26
                          Gambia"       0.72      0.81      0.76        36
           Saint Kitts and Nevis"       1.00      1.00      1.00        20
                        Mongolia"       0.96      0.96      0.96        23
                      Madagascar"       1.00      0.91      0.96        35
                          Poland"       0.92      0.96      0.94        23
                             USA"       0.94      0.94      0.94        17
                        Dominica"       0.94      0.97      0.95        31
                         Andorra"       0.98      0.91      0.94        53
                      Mauritania"       0.92      0.96      0.94        24
                        Slovakia"       0.85      1.00      0.92        22
                         Somalia"       1.00      0.94      0.97        16
                        Paraguay"       1.00      0.96      0.98        25
                          Sweden"       1.00      0.93      0.96        27
                          Malawi"       1.00      1.00      1.00        19
                         Finland"       0.40      0.94      0.57        18
                            Mali"       0.00      0.00      0.00        24
                          Norway"       1.00      0.83      0.91        18
                         Senegal"       0.90      1.00      0.95        18
                          Serbia"       1.00      0.94      0.97        18
             Antigua and Barbuda"       1.00      0.96      0.98        25
                          Brunei"       0.89      0.97      0.93        35
                            Togo"       1.00      0.95      0.97        19
                           Chile"       0.86      0.86      0.86         7
                          Rwanda"       0.83      1.00      0.91        24
                            Iran"       0.94      0.89      0.92        19
                         Croatia"       0.85      0.94      0.89        31
                     Ivory Coast"       1.00      0.82      0.90        17
                         Bahrain"       1.00      0.91      0.95        33
                     South Korea"       1.00      1.00      1.00        21
                 North Macedonia"       0.95      0.95      0.95        20
                           Japan"       0.95      1.00      0.97        19
Saint Vincent and the Grenadines"       0.93      0.96      0.95        27
                         Romania"       1.00      1.00      1.00        20
                      Cape Verde"       0.93      0.89      0.91        28
                       Luxemborg"       0.79      1.00      0.88        30
                     Afghanistan"       0.96      0.96      0.96        28
                         Uruguay"       0.93      1.00      0.96        27
                     El Salvador"       1.00      1.00      1.00        29
                         Armenia"       0.73      0.90      0.81        21
                            Chad"       0.95      1.00      0.97        39
                      Uzbekistan"       1.00      0.95      0.97        19
                       Lithuania"       0.91      0.95      0.93        21
               Equatorial Guinea"       0.51      0.87      0.65        23
                         Iceland"       0.88      1.00      0.94        30
                       Venezuela"       0.91      0.91      0.91        22
                         Modolva"       0.89      1.00      0.94        25
                         Morocco"       0.87      0.95      0.91        21
                        Thailand"       0.82      0.88      0.85        26
                        Colombia"       0.82      1.00      0.90        33
                     Timor Leste"       0.92      1.00      0.96        36
                          Zambia"       0.95      0.91      0.93        22
                     South Sudan"       0.95      0.91      0.93        23
                  United Kingdom"       1.00      0.65      0.78        31
                      Mozambique"       0.93      1.00      0.97        14
             St Kittis and Nevis"       0.00      0.00      0.00         6
                            Peru"       0.71      1.00      0.83        15
                         Grenada"       0.89      0.89      0.89        19
                          Panama"       1.00      0.92      0.96        24
                  Czech Republic"       0.97      1.00      0.98        28
                             CAR"       1.00      1.00      1.00        24
                        Suriname"       1.00      0.91      0.95        23
                         Austria"       0.92      0.87      0.89        38
                         Belgium"       1.00      0.90      0.95        21
                    Liechenstein"       0.93      1.00      0.96        27
                          Israel"       0.96      1.00      0.98        24
                           Malta"       0.95      0.83      0.89        24
                Marshall Islands"       0.92      0.86      0.89        14
                    Turkmenistan"       1.00      0.98      0.99        42
                          Latvia"       0.88      0.93      0.90        15
                        Tanzania"       0.97      0.97      0.97        35
                         Liberia"       1.00      0.86      0.93        22
                        Botswana"       0.86      0.89      0.88        28
              Dominican Republic"       1.00      0.90      0.95        29
                          Brazil"       1.00      1.00      1.00        19
                      Tajikistan"       0.95      0.95      0.95        20
                         Bahamas"       0.96      0.83      0.89        30
                    Doctor Congo"       0.93      0.87      0.90        30
                          Jordan"       1.00      1.00      1.00        13
                           Libya"       0.67      0.18      0.29        22
                         Nigeria"       0.95      1.00      0.97        19
                           Syria"       1.00      0.94      0.97        35
                           Nepal"       0.96      1.00      0.98        22
                          Guyana"       0.95      0.95      0.95        22
                        Portugal"       0.91      0.95      0.93        22
                     North Korea"       0.95      1.00      0.98        21
                    Burkina Faso"       1.00      0.96      0.98        24
                         Tunisia"       1.00      0.94      0.97        33
                          Cyprus"       0.84      0.95      0.89        22
                     Philippines"       0.83      1.00      0.91        10
                           Egypt"       0.83      1.00      0.91        25
                        Ethiopia"       0.96      1.00      0.98        26
                         Lebanon"       0.94      0.85      0.89        20
                        Djibouti"       1.00      0.88      0.93        16
                         Comoros"       0.85      0.89      0.87        19
             Trinidad and Tobago"       0.96      0.96      0.96        23
                      Seychelles"       1.00      0.90      0.95        21
                           Tonga"       0.62      1.00      0.76        24
                         Vietnam"       0.81      1.00      0.89        21
                      Montenegro"       0.93      0.93      0.93        14
                            Fiji"       1.00      1.00      1.00        30
                       Sri Lanka"       0.76      0.87      0.81        15
                         Burundi"       1.00      1.00      1.00        34
                           Qatar"       0.93      0.96      0.95        28
                           Spain"       0.96      1.00      0.98        22
                      Kyrgyzstan"       0.86      1.00      0.92        30
                          Russia"       0.69      0.56      0.62        32
                      San Marino"       0.95      1.00      0.97        18
                         Eritrea"       0.93      0.90      0.91        29
                        Barbados"       0.45      0.50      0.47        18
                        Zimbabwe"       0.92      0.83      0.87        29
                           Samoa"       1.00      1.00      1.00        31
                           India"       1.00      0.96      0.98        24
                             UAE"       0.91      1.00      0.95        30
                          Mexico"       0.83      0.94      0.88        16
                      Bangladesh"       1.00      0.95      0.98        22
                         Georgia"       1.00      0.89      0.94        18
                            Iraq"       0.95      0.91      0.93        22
                           Palau"       0.96      0.96      0.96        26
                         Algeria"       1.00      0.83      0.91        24
                          Canada"       1.00      0.92      0.96        26
                          France"       1.00      0.94      0.97        16
                        Cameroon"       1.00      0.90      0.95        21
                           Congo"       0.94      1.00      0.97        16
                      Costa Rica"       0.84      0.87      0.85        30
                       Argentina"       0.90      0.95      0.92        19
                Papua New Guinea"       0.89      0.80      0.84        30
                         Albania"       1.00      0.88      0.93        24
                          Bhutan"       0.90      0.96      0.93        27
                         Estonia"       1.00      1.00      1.00        37
                        Slovenia"       0.50      0.45      0.48        22
                          Belize"       0.96      1.00      0.98        27
                         Vanuatu"       1.00      0.95      0.98        21
                       Australia"       1.00      1.00      1.00        24
                         Jamaica"       0.92      1.00      0.96        36
                           Niger"       1.00      1.00      1.00        27
                        Pakistan"       1.00      0.91      0.96        35
                      Micronesia"       0.96      0.93      0.95        29
                       Indonesia"       0.92      0.92      0.92        24
                            Cuba"       1.00      1.00      1.00        30
                     Netherlands"       1.00      0.93      0.97        30
                       Mauritius"       1.00      0.94      0.97        17
                      Kazakhstan"       0.97      0.97      0.97        33
                      Azerbaijan"       1.00      0.82      0.90        22
                           Ghana"       1.00      0.97      0.98        29
                           China"       0.88      0.88      0.88        24
                            Oman"       0.89      0.94      0.91        17
           Sao Tome and Principe"       1.00      1.00      1.00        21
                           Yemen"       0.90      0.86      0.88        21
                         Germany"       1.00      1.00      1.00        33
                          Turkey"       1.00      0.75      0.86        20
                        Bulgaria"       0.94      0.85      0.89        20
                           Gabon"       1.00      1.00      1.00        24
                        Honduras"       0.93      0.90      0.92        30
                        Cambodia"       0.84      0.91      0.87        23
                     Saint Lucia"       0.94      1.00      0.97        17
                    Saudi Arabia"       0.94      0.94      0.94        18
                         Ukraine"       0.89      0.81      0.85        21
                           Sudan"       1.00      0.94      0.97        18
                         Bolivia       1.00      0.97      0.98        29]
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
