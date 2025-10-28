from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# ✅ Load trained model
model = load_model("plant_disease_cnn_model.h5")

# ✅ Class labels (same order as training)
class_labels = {
    0: 'Pepper__bell___Bacterial_spot',
    1: 'Pepper__bell___healthy',
    2: 'Potato___Early_blight',
    3: 'Potato___healthy',
    4: 'Potato___Late_blight',
    5: 'Tomato___Target_Spot',
    6: 'Tomato___Tomato_mosaic_virus',
    7: 'Tomato___Tomato_YellowLeaf_Curl_Virus',
    8: 'Tomato___Bacterial_spot',
    9: 'Tomato___Early_blight',
    10: 'Tomato___healthy',
    11: 'Tomato___Late_blight',
    12: 'Tomato___Leaf_Mold',
    13: 'Tomato___Septoria_leaf_spot',
    14: 'Tomato___Spider_mites_Two_spotted_spider_mite'
}

# ✅ Home route (upload form)
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# ✅ Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400
    file = request.files['file']

    # Save image temporarily
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Load and preprocess image
    img = image.load_img(file_path, target_size=(64, 64))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    class_index = np.argmax(pred, axis=1)[0]
    predicted_class = class_labels[class_index]

    return render_template('index.html', 
                           prediction=predicted_class, 
                           img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
