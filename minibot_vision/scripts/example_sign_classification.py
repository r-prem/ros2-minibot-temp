#!/usr/bin/env python3

import cv2
import TmClassification
#from PIL import Image, ImageOps
import numpy as np
from copy import deepcopy

# ROS
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# *** hyper params ***
URL = 'https://teachablemachine.withgoogle.com/models/3dHrDR9Aq/'
IMG_RES = (480, 640)
TF_RES = (224, 224)     # tf is cropping the image

# *** globals ***
img_stream = np.zeros((IMG_RES[0], IMG_RES[1], 3), np.uint8)
tf = TmClassification.TmClassification()
#tf = TmClassification.TmClassification(URL)
cv_bridge = CvBridge()


def image_color_callback(img_ros):
    global img_stream, cv_bridge, tf
    # convert to cv2
    try:
        img_color = cv_bridge.imgmsg_to_cv2(img_ros, "bgr8")

        prediction, probability = tf.predictImage(img_color)
        cv2.putText(img_color, "c: {} p: {}".format(prediction, probability), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
        cv2.rectangle(img_color, (IMG_RES[1] // 2 - (TF_RES[1] // 2), IMG_RES[0] // 2 - (TF_RES[0] // 2)), (IMG_RES[1] // 2 + (TF_RES[1] // 2), IMG_RES[0] // 2 + (TF_RES[0] // 2)), (255, 255, 255), thickness=2)
        img_stream = deepcopy(img_color)
    except CvBridgeError as e:
        print(e)

if __name__ == "__main__":
    rospy.init_node("example_sign_classification")

    # get camera stream
    img_color_topic = "/camera/color/image_raw"
    image_depth_sub = rospy.Subscriber(img_color_topic, Image, image_color_callback)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        cv2.imshow("IMG_Color", img_stream)
        cv2.waitKey(1)

        rate.sleep()

    rospy.loginfo("Node is shutting down. Closing all cv2 windows...")
    cv2.destroyAllWindows()
