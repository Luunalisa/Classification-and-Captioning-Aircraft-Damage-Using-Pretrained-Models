import tensorflow as tf
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

#load the pretrained BLIP processor and model:
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, processor, model, **kwargs):
        """
        Initialize the custom Keras layer with the BLIP processor and model.

        Args:
            processor: The BLIP processor for preparing inputs for the model.
            model: The BLIP model for generating captions or summaries.
        """
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        # Use tf.py_function to run the custom image processing and text generation
        return tf.py_function(self.process_image, [image_path, task], tf.string)

    def process_image(self, image_path, task):
        """
        Perform image loading, preprocessing, and text generation.
        """
        image_path_str = image_path.numpy().decode("utf-8")
        image = Image.open(image_path_str).convert("RGB")

        if task.numpy().decode("utf-8") == "caption":
            prompt = "This is a picture of"
        else:
            prompt = "This is a detailed photo showing"

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        output = self.model.generate(**inputs)

        result = self.processor.decode(output[0], skip_special_tokens=True)
        return result

def generate_text(image_path, task):
    blip_layer = BlipCaptionSummaryLayer(processor, model)
    return blip_layer(image_path, task)