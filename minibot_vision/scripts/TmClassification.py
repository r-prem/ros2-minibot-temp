from select import select
import requests
import os

import rclpy
from keras.models import load_model
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json
from rosidl_runtime_py import get_interface_path
import shutil


class TmClassification:
    files = ['model.json', 'metadata.json', 'model.weights.bin']
    tfjs_dir = "/resources/tfjs_model"
    h5_dir = "/resources/h5_model"
    h5_file = "model.h5"

    def __init__(self, url=None):
        ros_dir = get_interface_path("minibot_vision")
        self.tfjs_dir = ros_dir + self.tfjs_dir
        self.h5_dir = ros_dir + self.h5_dir

        if url is not None:
            self.setNewModel(url)

        self.loadNewModel()
        print("TF: ready, waiting for images to classify.")

        if not tf.test.is_built_with_cuda():
            print("Your tf build has no CUDA support.")

    def setNewModel(self, url):
        print("TF: Downloading model from url: {}".format(url))
        self._prepareDirectories()
        self._downloadFiles(url)
        self._convertFromFfjsFoKeras()

        self.loadNewModel()

    def loadNewModel(self):
        # TODO if there is an existing model, else error
        if not os.path.exists(f'{self.tfjs_dir}/{self.files[1]}') or not os.path.exists(f'{self.h5_dir}/{self.h5_file}'):
            #rospy.logwarn("({}) There is no existing tensorflow model on your machine. You can set a new model by calling the /set_model service.".format(rospy.get_name()))
            return
        # Load the model
        self.model = load_model(f'{self.h5_dir}/{self.h5_file}', compile=False)
        # Load metadata for labels
        self.metadata = self._loadMetadata()
        self._uploadLabelsToParamServer()

    def _prepareDirectories(self):
        if os.path.exists(self.tfjs_dir):
            shutil.rmtree(self.tfjs_dir)
        if os.path.exists(self.h5_dir):
            shutil.rmtree(self.h5_dir)

        os.mkdir(self.h5_dir)
        os.mkdir(self.tfjs_dir)

    def _downloadFiles(self, url):
        for f in self.files:
            request_url = url + f
            storage_file = f'{self.tfjs_dir}/{f}'
            r = requests.get(request_url, allow_redirects=True)
            open(storage_file, 'wb').write(r.content)

    def _convertFromFfjsFoKeras(self):
        os.system(f'tensorflowjs_converter --input_format=tfjs_layers_model --output_format=keras {self.tfjs_dir}/{self.files[0]} {self.h5_dir}/{self.h5_file}')

    def _loadMetadata(self):
        f = open(self.tfjs_dir+'/'+self.files[1])
        return json.load(f)

    def predictImage(self, image):
        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Replace this with the path to your image
        #image = Image.open('rosa.jpg')
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center

        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = self.model.predict(data)
        
        # Generate arg maxes of predictions
        class_nr = np.argmax(prediction, axis=1)[0]
        return class_nr, np.max(prediction, axis=1)[0]

    def _uploadLabelsToParamServer(self):
        # delete all existing params in this namespace
        try:
            rospy.delete_param('sign_classification/class_labels/')
        except KeyError:
            pass

        labels = self.metadata["labels"]
        for i, (l) in enumerate(labels):
            rospy.set_param("sign_classification/class_labels/{}".format(i), l)

    def labelOfClass(self, class_number):
        labels = self.metadata["labels"]
        if class_number < 0 or class_number > len(labels):
            return 'unkown'
        return labels[class_number]
    