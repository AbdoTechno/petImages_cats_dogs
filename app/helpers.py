import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

CLASSES = ["Cat", "Dog"]


def load_model(path):

    interpreter = tf.lite.Interpreter(model_path=path)

    interpreter.allocate_tensors()

    return interpreter


def prepare_image(image_file, target_size):

    image = Image.open(image_file).convert("RGB")

    image = image.resize(target_size)

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def get_prediction(model, image_file, model_type):

    input_details = model.get_input_details()

    input_shape = input_details[0]["shape"]

    # Example:
    # [1, 160, 160, 3]
    # [1, 224, 224, 3]

    height = input_shape[1]
    width = input_shape[2]

    image_array = prepare_image(image_file, (width, height))

    # =========================
    # Preprocessing
    # =========================

    if model_type == "cnn":

        image_array = image_array / 255.0

    elif model_type == "vgg":

        image_array = preprocess_input(image_array)

    else:

        raise ValueError("model_type must be 'cnn' or 'vgg'")

    # =========================
    # Inference
    # =========================

    model.set_tensor(input_details[0]["index"], image_array)

    model.invoke()

    output_details = model.get_output_details()

    output_data = model.get_tensor(output_details[0]["index"])

    probabilities = output_data[0]

    prediction_index = np.argmax(probabilities)

    prediction = CLASSES[prediction_index]

    confidence = probabilities[prediction_index]

    return prediction, confidence
