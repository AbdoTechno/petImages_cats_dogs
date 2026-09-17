import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

classes = ['Cat', 'Dog']
def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def load_and_preprocess_image(image_file):
    img = load_img(image_file, target_size=(224, 224))
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def get_prediction(model, image_file):
    img_array = load_and_preprocess_image(image_file)
    model.set_tensor(model.get_input_details()[0]["index"], img_array)
    model.invoke()
    output_data = model.get_tensor(model.get_output_details()[0]["index"])
    return classes[np.argmax(output_data[0])] 

