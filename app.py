from flask import Flask, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)


model_path = 'fashion_mnist_model.h5'  
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fashion MNIST class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def normalize_image(image_path):
    """Preprocess image for prediction"""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        return img_array
    except Exception as e:
        print(f"Error in normalize_image: {str(e)}")
        return None

@app.route('/')
def index():
    return "Simple API to classify Fashion MNIST images using Flask!"

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Initialize an empty list to store predictions
        predictions = []

        # Directory path containing multiple images
        image_dir = 'new_data'  # Path to your image folder
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(('.jpg', '.png'))]  # Include multiple image types

        # Check if there are no images to process
        if not image_paths:
            return jsonify({'message': 'No new data to process'})

        # Process each image in the batch
        for image_path in image_paths:
            normalized_image = normalize_image(image_path)

            if normalized_image is not None:
                # Make predictions using the pre-loaded model
                prediction = model.predict(normalized_image)
                print(f"Raw predictions for {image_path}: {prediction}")

                # Interpret the result and append it to the predictions list
                predicted_class = class_names[np.argmax(prediction)]
                predictions.append({'image': image_path, 'prediction': predicted_class})
            else:
                print(f"Image normalization failed for {image_path}")

        return jsonify({'predictions': predictions})

    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app in production mode (no debug)
    app.run(host='0.0.0.0', port=5000, debug=False)
