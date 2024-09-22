import requests
import os

# The API endpoint
api_url = "http://127.0.0.1:5000/predict"

# Directory containing images
image_dir = 'new_data'
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

if image_paths:
    response = requests.post(api_url, json={'image_paths': image_paths})
    predictions = response.json().get('predictions', [])

    # Save or print batch results
    for prediction in predictions:
        print(prediction)
else:
    print('No images found in new_data directory.')
