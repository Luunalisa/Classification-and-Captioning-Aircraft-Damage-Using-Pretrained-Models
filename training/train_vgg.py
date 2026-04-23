import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import batch_size, n_epochs, img_rows, img_cols, extracted_folder
from models.vgg_classifier import get_model

# Set seed for reproducibility
def train_model():
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Define directories for train, test, and validation splits
    extract_path = extracted_folder
    train_dir = os.path.join(extract_path, 'train')
    valid_dir = os.path.join(extract_path, 'valid')

    # Create ImageDataGenerators to preprocess the data
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_rows, img_cols),   # Resize images to the size VGG16 expects
        batch_size=batch_size,
        seed = seed_value,
        class_mode='binary',
        shuffle=True # Binary classification: dent vs crack
    )

    valid_generator =  valid_datagen.flow_from_directory(
        directory= valid_dir,
        class_mode= 'binary',
        seed=seed_value,
        batch_size=batch_size,
        shuffle=False,
        target_size=(img_rows, img_cols)
    )

    model = get_model(img_rows, img_cols)

    history = model.fit(
        train_generator ,  
        epochs=n_epochs,  
        validation_data=valid_generator, 
    )

    return model, history