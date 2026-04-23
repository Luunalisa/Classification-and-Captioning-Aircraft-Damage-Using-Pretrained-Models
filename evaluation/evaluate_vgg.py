from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import batch_size, img_rows, img_cols, extracted_folder
import os

# Evaluate the model on the test set
def evaluate_model(model):

    extract_path = extracted_folder
    test_dir = os.path.join(extract_path, 'test')

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator =  test_datagen.flow_from_directory(
        directory= test_dir,
        class_mode= 'binary',
        batch_size=batch_size,
        shuffle=False,
        target_size=(img_rows, img_cols)
    )

    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_generator