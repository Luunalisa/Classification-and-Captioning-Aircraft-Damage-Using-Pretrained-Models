import tensorflow as tf
import matplotlib.pyplot as plt
from models.blip_captioning import generate_text

def run_caption(image_url):

    # Load and display the image
    img = plt.imread(image_url)
    plt.imshow(img)
    plt.axis('off')  # Hide the axis
    plt.show()

    image_path = tf.constant(image_url)

    # Generate a caption for the image
    caption = generate_text(image_path, tf.constant("caption"))
    print("Caption:", caption.numpy().decode("utf-8"))

    # Generate a summary for the image
    summary = generate_text(image_path, tf.constant("summary"))
    print("Summary:", summary.numpy().decode("utf-8"))