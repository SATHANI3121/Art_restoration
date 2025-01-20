import numpy as np
from similarity import initialize_model, extract_features, load_csv_data
import os

# Load the VGG16 model
model = initialize_model()

# Load image paths and details from the CSV
csv_file_path = 'final.csv'
image_folder = 'zip'
image_paths, details = load_csv_data(csv_file_path, image_folder)

# Extract features for all images in the CSV file
features_list = [extract_features(model, img_path) for img_path in image_paths]

# Save the features and image paths
np.save('precomputed_features.npy', np.array(features_list))
np.save('image_paths.npy', np.array(image_paths))
np.save('details.npy', np.array(details))
