import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

# Load the CSV file
csv_file_path = 'final.csv'
df = pd.read_csv(csv_file_path)

# Print column names to verify them
print(df.columns)

# Replace these with actual column names from your CSV
image_path_column = 'Path to image'  # This should be the column name that contains the image file names
details_column = 'Details of the statue'  # This should be the column name that contains the details

# Extract image paths and details
image_paths = df[image_path_column].tolist()
details = df[details_column].tolist()

# Folder containing the images
image_folder = 'zip'

# Function to create full image paths
def get_full_image_path(relative_path):
    return os.path.join(image_folder, relative_path)

# Update image paths to be full paths
image_paths = [get_full_image_path(p) for p in image_paths]

# Load the VGG16 model and remove the top layer
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Function to preprocess image and extract features
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Extract features for images in the CSV file
features_list = []
for img_path in image_paths:
    features = extract_features(img_path)
    features_list.append(features)

# Function to find the most similar images
def find_similar_images(user_img_path, features_list, image_paths, details, top_n=5, threshold=0.8):
    user_features = extract_features(user_img_path)
    similarities = cosine_similarity([user_features], features_list)
    similar_indices = [i for i in range(len(similarities[0])) if similarities[0][i] >= threshold]
    similar_indices = sorted(similar_indices, key=lambda i: similarities[0][i], reverse=True)[:top_n]
    similar_images = [(image_paths[i], details[i], similarities[0][i]) for i in similar_indices]
    return similar_images

# Function to display similar images
def display_similar_images(similar_images):
    for img_path, detail, similarity in similar_images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f'Similarity: {similarity:.2f}\nDetails: {detail}')
        plt.axis('off')
        plt.show()

# Example usage
user_img_path = 'zip/Slightly Broken/scul30.jpg'
similar_images = find_similar_images(user_img_path, features_list, image_paths, details)
display_similar_images(similar_images)
