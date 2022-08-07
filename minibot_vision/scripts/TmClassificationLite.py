import tflite_runtime.interpreter as tflite
from PIL import Image, ImageOps
import numpy as np
from rosidl_runtime_py import get_interface_path
import csv


class TmClassificationLite:
    file_names = ['model_unquant.tflite', 'labels.txt']
    model_dir = '/resources/tf_lite_model/'

    def __init__(self):
        #rospack = rospkg.RosPack()
        self.model_dir = get_interface_path("minibot_vision") + self.model_dir

        # Load the TFLite model in TFLite Interpreter
        self.interpreter = tflite.Interpreter(self.model_dir + self.file_names[0])
        self.interpreter.allocate_tensors()

        self.labels = self.read_labels()

    def read_labels(self):
        # This is truly not a nice implementation... But it works!
        reader = csv.reader(open(self.model_dir + self.file_names[1]), delimiter=' ')
        result = []
        for s in reader:
            result.append(s[1])

        return result


    def preprocess_image(self, image):
        img_pil = Image.fromarray(image)
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Replace this with the path to your image
        #image = Image.open('rosa.jpg')
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(img_pil, size, Image.ANTIALIAS, centering=(0.5, 0.5))

        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array

        return data

    def predict_image(self, img):
        img_preprocessed = self.preprocess_image(img)

        input_details = self.interpreter.get_input_details()
        self.interpreter.set_tensor(input_details[0]['index'], img_preprocessed)
        self.interpreter.invoke()

        output_details = self.interpreter.get_output_details()
        output = self.interpreter.get_tensor(output_details[0]['index'])

        return output, self.labels
