from flask import Flask, render_template, request
import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
from tensorflow.keras.models import load_model

# Create Flask app instance
app = Flask(__name__)

# Define the function to predict the class
def pred_dental_problem(model, dental_image):
    try:
        test_image = load_img(dental_image)
        print("@@ Got Image for prediction")

        # Resize the image to (128, 128)
        test_image = resize(test_image, (128, 128))

        test_image = img_to_array(test_image) / 255
        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image)
        print('@@ Raw result = ', result)

        pred = np.argmax(result, axis=1)[0]
        print("Prediction Index:", pred)

        classes = ["abscess", "impacted tooth", "tooth decay"]
        pred_label = classes[pred]

        return pred_label, f'{pred_label}.html'

    except Exception as e:
        print("Error during prediction:", e)
        return "Error", 'error.html'


# Load the model
model_path = r'E:\local disk D\New folder (2)\dental problem detection\my_model.h5'
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        print("Model Loaded Successfully")
    except Exception as e:
        print("Error loading model:", e)
        exit()
else:
    print("Model file not found at:", model_path)
    exit()

# Define routes
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('E:/local disk D/New folder (2)/dental problem detection/static/upload', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_dental_problem(model, dental_image=file_path)

        # Construct path to the template file
        output_page_path = os.path.join('templates', output_page)

        return render_template(output_page_path, pred_output=pred, user_image=file_path)

# Run the Flask app
if __name__ == "__main__":
    app.run(threaded=False, port=8080)
