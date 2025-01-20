from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import hello

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'
app.config['OUTPUT_FOLDER'] = 'static/output/'

# Ensure the upload, processed, and output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/reconstruct', methods=['GET', 'POST'])
def reconstruct():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'original_image' not in request.files or 'processed_image' not in request.files:
            return redirect(request.url)

        original_image = request.files['original_image']
        processed_image = request.files['processed_image']

        # If user does not select file, browser also
        # submit an empty part without filename
        if original_image.filename == '' or processed_image.filename == '':
            return redirect(request.url)

        if original_image and processed_image:
            original_filename = secure_filename(original_image.filename)
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            original_image.save(original_image_path)

            processed_filename = secure_filename(processed_image.filename)
            processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            processed_image.save(processed_image_path)

            # Call hello.py to process the images
            output_dir = app.config['OUTPUT_FOLDER']
            hello.process_images(original_image_path, processed_image_path, output_dir)

            # Get paths of output images
            output_images = []
            for filename in os.listdir(output_dir):
                if filename.endswith('.png'):
                    output_images.append(os.path.join(app.config['OUTPUT_FOLDER'], filename))

            return render_template('result.html', output_images=output_images)

    return render_template('reconstruct.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
