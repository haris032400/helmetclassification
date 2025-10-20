from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model('Downloads/Project3/helmet_model.h5')
IMG_SIZE = 128
CLASS_NAMES = ['Dont Wear Helmet', 'Wear Helmet']  # Update if needed

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = float(predictions[0][class_index])

    return CLASS_NAMES[class_index], confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    uploaded_image = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, conf = predict_image(filepath)
            prediction = label
            confidence = round(conf * 100, 2)
            uploaded_image = filename

    return render_template('index.html', prediction=prediction, confidence=confidence, uploaded_image=uploaded_image)

if __name__ == '__main__':
    app.run(debug=True)
