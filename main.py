from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from webpages import home_page, retina_upload_page, derma_upload_page
import numpy as np
import cv2
from tensorflow import keras
import os


UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# model locations
retinamnist_model = "retinamnist_model"
dermamnist_model = "dermamnist_model"

retina_labels = {0: 'class0',
                1: 'class1',
                2: 'class2',
                3: 'class3',
                4: 'class4',
                5: 'class5',
                6: 'class6'}
derma_labels = {0: 'actinic keratoses and intraepithelial carcinoma',
          1: 'basal cell carcinoma',
          2: 'benign keratosis-like lesions',
          3: 'dermatofibroma',
          4: 'melanoma',
          5: 'melanocytic nevi',
          6: 'vascular lesions'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def base():
    return home_page


@app.route('/retina', methods=['GET', 'POST'])
def retina_model():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "image.png"))
            image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER']+'image.png'))
            image = cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)
            model = keras.models.load_model(retinamnist_model)
            numpy_data = np.array([np.asarray(image)])
            y_proba = model.predict(numpy_data/255)
            y_pred = np.argmax(y_proba, axis=-1)
            return '''
                <!DOCTYPE html>
                <head>
                <title>MedMNIST-Retina</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
                </head>
                <body>

                <div class="jumbotron text-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" fill="currentColor" class="bi bi-eye-fill" viewBox="0 0 16 16">
                    <path d="M10.5 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/>
                    <path d="M0 8s3-5.5 8-5.5S16 8 16 8s-3 5.5-8 5.5S0 8 0 8zm8 3.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/>
                </svg>
                <h1>'''+str(retina_labels[y_pred[0]])+'''</h1>
                </div>
                <div class="container text-center">
                <a href="http://127.0.0.1:5000/"><p class="btn btn-default">Home</p></a>
                </div>

                </body>
                </html>
             '''
    return retina_upload_page

@app.route('/derma', methods=['GET', 'POST'])
def derma_model():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'image.png'))
            image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER']+'image.png'))
            image = cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA)
            model = keras.models.load_model(dermamnist_model)
            numpy_data = np.array([np.asarray(image)])
            y_proba = model.predict(numpy_data/255)
            y_pred = np.argmax(y_proba, axis=-1)
            return '''
                <!DOCTYPE html>
                <head>
                <title>MedMNIST-Derma</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
                </head>
                <body>

                <div class="jumbotron text-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" fill="currentColor" class="bi bi-bandaid-fill" viewBox="0 0 16 16">
                    <path d="m2.68 7.676 6.49-6.504a4 4 0 0 1 5.66 5.653l-1.477 1.529-5.006 5.006-1.523 1.472a4 4 0 0 1-5.653-5.66l.001-.002 1.505-1.492.001-.002Zm5.71-2.858a.5.5 0 1 0-.708.707.5.5 0 0 0 .707-.707ZM6.974 6.939a.5.5 0 1 0-.707-.707.5.5 0 0 0 .707.707ZM5.56 8.354a.5.5 0 1 0-.707-.708.5.5 0 0 0 .707.708Zm2.828 2.828a.5.5 0 1 0-.707-.707.5.5 0 0 0 .707.707Zm1.414-2.121a.5.5 0 1 0-.707.707.5.5 0 0 0 .707-.707Zm1.414-.707a.5.5 0 1 0-.706-.708.5.5 0 0 0 .707.708Zm-4.242.707a.5.5 0 1 0-.707.707.5.5 0 0 0 .707-.707Zm1.414-.707a.5.5 0 1 0-.707-.708.5.5 0 0 0 .707.708Zm1.414-2.122a.5.5 0 1 0-.707.707.5.5 0 0 0 .707-.707ZM8.646 3.354l4 4 .708-.708-4-4-.708.708Zm-1.292 9.292-4-4-.708.708 4 4 .708-.708Z"/>
                </svg>
                <h1>'''+str(derma_labels[y_pred[0]])+'''</h1>
                </div>
                <div class="container text-center">
                <a href="http://127.0.0.1:5000/"><p class="btn btn-default">Home</p></a>
                </div>

                </body>
                </html>
             '''
    return derma_upload_page

app.run(debug = True)