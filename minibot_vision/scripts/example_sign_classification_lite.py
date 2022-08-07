#!/usr/bin/env python3

import TmClassificationLite
import numpy as np
import cv2
from copy import deepcopy

# ROS
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# *** hyper params ***
IMG_RES = (480, 640)
TF_RES = (224, 224)     # tf is cropping the image

# *** globals ***
model = TmClassificationLite.TmClassificationLite()
img_stream = np.zeros((IMG_RES[0], IMG_RES[1], 3), np.uint8)
cv_bridge = CvBridge()

def image_color_callback(img_ros):
    global img_stream, cv_bridge, model
    # convert to cv2
    try:
        # convert image
        img_color = cv_bridge.imgmsg_to_cv2(img_ros, "bgr8")

        # call model
        predictions, labels = model.predict_image(img_color)
        max_class_nr = np.argmax(predictions, axis=1)[0]
        max_prediction = np.max(predictions, axis=1)[0]

        # visualize
        cv2.putText(img_color, "c: {} p: {}".format(labels[max_class_nr], max_prediction), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
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
