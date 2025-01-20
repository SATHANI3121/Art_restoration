import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load the VGG16 model and remove the top layer
def initialize_model():
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    return model

# Preprocess image and extract features
def extract_features(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Find similar images based on cosine similarity
def find_similar_images(features, precomputed_features, image_paths, details, top_n=5, threshold=0.8):
    similarities = cosine_similarity([features], precomputed_features)
    similar_indices = [i for i in range(len(similarities[0])) if similarities[0][i] >= threshold]
    similar_indices = sorted(similar_indices, key=lambda i: similarities[0][i], reverse=True)[:top_n]
    similar_images = [(image_paths[i], details[i], similarities[0][i]) for i in similar_indices]
    return similar_images

# Load CSV data
def load_csv_data(csv_file_path, image_folder):
    df = pd.read_csv(csv_file_path)
    image_paths = df['Path to image'].tolist()
    details = df['Details of the statue'].tolist()
    image_paths = [os.path.join(image_folder, p) for p in image_paths]
    return image_paths, details
