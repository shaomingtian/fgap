import numpy as np
import pandas as pd
import argparse
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Flatten
import os

import tensorflow as tf
import random

import fgap_config
from logger_module import log_message

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Define a function to build the new model
def build_feature_extractor(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add a fully connected layer for dimensionality reduction
    flatten_layer = Flatten()(base_model.output)
    dense_layer = Dense(256, activation='relu')(flatten_layer)

    # Create a new model
    model = Model(inputs=base_model.input, outputs=dense_layer)
    return model

# Load the feature extraction model
model = build_feature_extractor((224, 224, 3))

# Preprocess image function
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # VGG16 input size is 224x224
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add a dimension
    img_array = preprocess_input(img_array)  # Preprocess the input
    return img_array

# Feature extraction function
def extract_features(img_path):
    img_array = load_and_preprocess_image(img_path)
    features = model.predict(img_array)  # Perform feature extraction
    return features.flatten()  # Flatten the feature array for storage

# Update Excel with image features
def update_excel_with_features(excel_path, image_name, features):
    # Read Excel
    df = pd.read_excel(excel_path)

    # Get the filename
    row_filename = os.path.basename(image_name).rstrip(".png")
    # TODO: may error when there are same filename in different dirs
    row_index = df[df.iloc[:, 1] == row_filename].index

    if not row_index.empty:
        index = row_index[0]  # Find the filename (first)
        required_columns = 22 + len(features)

        # Ensure DataFrame has enough columns to store features
        current_columns = df.shape[1]

        if current_columns < required_columns:
            # Create new columns with NaN
            new_columns = {f'Image_Feature_{i}': [np.nan] * df.shape[0] for i in range(current_columns - 22, required_columns - 22)}
            df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

        # Write features from column 22
        for i in range(len(features)):
            df.iloc[index, 22 + i] = features[i]

        # Save Excel
        df.to_excel(excel_path, index=False)
        log_message(f"Successfully updated features for '{image_name}' in row {index + 1} of the Excel file.")
    else:
        log_message(f"Image name '{image_name}' not found in the Excel file.")

# Main function
def get_image_feature_to_excel(img_path, excel_path):
    image_name = os.path.basename(img_path)  # Get image filename
    features = extract_features(img_path)  # Extract features
    # delete image or not
    if not fgap_config.IMAGE_KEEP:
        try:
            # check if exist
            if os.path.exists(img_path):
                os.remove(img_path)  # delete image
                log_message(f'File {img_path} has been deleted.')
            else:
                log_message(f'File {img_path} does not exist.')
        except Exception as e:
            log_message(f'Error occurred while trying to delete file: {e}')
    update_excel_with_features(excel_path, image_name, features)  # Update Excel

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Extract features from images and save to Excel.')
    parser.add_argument('image_paths', metavar='image_path', type=str, nargs='+',
                        help='List of image paths to extract features from.')
    parser.add_argument('output_xlsx', type=str,
                        help='Path to the Excel file to update with features.')

    args = parser.parse_args()

    # Ensure all image paths exist
    for img_path in args.image_paths:
        if not os.path.exists(img_path):
            log_message(f"Image path does not exist: {img_path}")
            exit(1)

    # Run the main function
    for img_path in args.image_paths:
        get_image_feature_to_excel(img_path, args.output_xlsx)