import requests
import os

# The API endpoint
api_url = "http://127.0.0.1:5000/predict"

# Directory containing images
image_dir = 'new_data'
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

if image_paths:
    files = []
    # Open each image file within the 'with' block to keep it open during the request
    for image_path in image_paths:
        files.append(('images', (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')))

    try:
        # Send the images to the Flask API
        response = requests.post(api_url, files=files)

        # Parse the response from the API
        if response.status_code == 200:
            predictions = response.json().get('predictions', [])
            for prediction in predictions:
                print(prediction)  # This will output the prediction to the console
else:
    print(f"Error: Received status code {response.status_code}")

            # Save or print batch results
            for prediction in predictions:
                print(prediction)
        else:
            print(f"Error: Received status code {response.status_code}")
    finally:
        # Close all the open files after the request is done
        for _, (filename, file, _) in files:
            file.close()
else:
    print('No images found in new_data directory.')
