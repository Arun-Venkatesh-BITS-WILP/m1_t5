import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# CIFAR-10 class labels
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", "dog",
    "frog", "horse", "ship", "truck"
]


def predict(img_path):
    # Load the trained model (saved as .h5 file)
    model = tf.keras.models.load_model("models/cnn_model.h5")

    # Preprocess the image
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.image.rgb_to_grayscale(img_array)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)

    # Get the class name corresponding to the predicted index
    predicted_class_name = class_names[predicted_class_index[0]]

    return predicted_class_name


if __name__ == "__main__":
    img_path = os.path.join("test_data", "airplane.jpg")
    print(f"Prediction: {predict(img_path)}")
