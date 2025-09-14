import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Step 2: Load and preprocess the dataset
dataset_path = '/content/drive/MyDrive/Anaemia/'

def load_images_from_folder(base_path, img_size=(128, 128)):
    images = []
    labels = []
    
    # Define class folders
    class_folders = {
        'Anaemic': 1,
        'Non-Anaemic': 0,
        'Non-Anrmic': 0  # Handle the typo in folder name
    }
    
    for class_name, label_id in class_folders.items():
        class_path = os.path.join(base_path, class_name)
        if os.path.exists(class_path):
            print(f"Loading images from {class_name} class...")
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                try:
                    # Check if it's an image file
                    if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                        # Read image
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize image
                            img = cv2.resize(img, img_size)
                            # Convert BGR to RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # Normalize pixel values
                            img = img / 255.0
                            images.append(img)
                            labels.append(label_id)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        else:
            print(f"Directory {class_path} does not exist.")
    
    return np.array(images), np.array(labels)

# Load images
X, y = load_images_from_folder(dataset_path)

print(f"Total images loaded: {len(X)}")
print(f"Number of anaemic images: {sum(y == 1)}")
print(f"Number of non-anaemic images: {sum(y == 0)}")

# Step 3: Explore the dataset
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X[i])
    plt.title(f"Label: {'Anaemic' if y[i] == 1 else 'Non-Anaemic'}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Check class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=y)
plt.title('Class Distribution')
plt.xlabel('Class (0: Non-Anaemic, 1: Anaemic)')
plt.ylabel('Count')
plt.show()

# Step 4: Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

# Step 5: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Fit the data generator
datagen.fit(X_train)