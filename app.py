import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import construct
import requests
import base64
from similarity import initialize_model, extract_features, find_similar_images



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'
app.config['OUTPUT_FOLDER'] = 'static/output/'

# Load the model once when the app starts
model = initialize_model()

# Load precomputed features, image paths, and details
precomputed_features = np.load('precomputed_features.npy')
image_paths = np.load('image_paths.npy')
details = np.load('details.npy')


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def manual_mask(input_image_path, output_image_path):
    global erasing, x_prev, y_prev, mask, image, radius

    erasing = False
    x_prev, y_prev = -1, -1
    radius = 10

    def erase(event, x, y, flags, param):
        global erasing, x_prev, y_prev, mask, image
        if event == cv2.EVENT_LBUTTONDOWN:
            erasing = True
            x_prev, y_prev = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if erasing:
                cv2.circle(mask, (x, y), radius, (0, 0, 0), -1)
                cv2.circle(image, (x, y), radius, (0, 0, 0), -1)
                cv2.imshow("Select Area", image)
        elif event == cv2.EVENT_LBUTTONUP:
            erasing = False

    image = cv2.imread(input_image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        return
    
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    cv2.imshow("Select Area", image)
    cv2.setMouseCallback("Select Area", erase)
    
    print("Press 'd' when done.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            break

    cv2.destroyAllWindows()

    new_image = np.zeros_like(image)
    new_image[mask == 0] = [255, 255, 255]
    
    cv2.imwrite(output_image_path, new_image)
    print(f"New image saved as '{output_image_path}'.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/explore', methods=['GET', 'POST'])
def explore():
    if request.method == 'GET':
        return render_template('explore.html')
    else:
        uploaded_file = request.files['statue_image']
        if uploaded_file.filename != '':
            file_path = os.path.join('static/uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Extract features for the uploaded image
            features = extract_features(model, file_path)

            # Find similar images
            similar_images = find_similar_images(features, precomputed_features, image_paths, details)

            return render_template('explore.html', similar_images=similar_images, features=features)
        return render_template('explore.html')

@app.route('/reconstruct', methods=['GET', 'POST'])
def reconstruct():
    if request.method == 'POST':
        if 'statue_image' not in request.files:
            return redirect(request.url)
        file = request.files['statue_image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + filename)
            manual_mask(file_path, processed_image_path)
            return render_template('reconstruct.html', original_image=file_path, processed_image=processed_image_path)
    return render_template('reconstruct.html')

@app.route('/result', methods=['POST'])
def result():
    original_image_path = request.form['original_image']
    processed_image_path = request.form['processed_image']
    output_dir = app.config['OUTPUT_FOLDER']

    print(original_image_path,processed_image_path)

    construct.process_images(original_image_path,processed_image_path, output_dir)

    output_images = []
    for filename in os.listdir(output_dir):
        if filename.startswith('output_') and filename.endswith('.png'):
            output_images.append(os.path.join(app.config['OUTPUT_FOLDER'], filename))

    return render_template('result.html', output_images=output_images)

if __name__ == '__main__':
    app.run(debug=True)

