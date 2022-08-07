#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import time
import rospkg
from pathlib import Path
from copy import copy
import SegmentSign
from std_srvs.srv import SetBool, SetBoolResponse

# *** hyper-params ***
IMG_RES = (480, 640)
TF_RES = (224, 224)     # our tensorflow is using a reduced image size
MAX_DURATION = 0.1      # max duration between screenshots in [sec]
IMAGE_NAME = ""
REMOTE_NODE = False
SAVE_DIR = "/resources/training_imgs/"

# *** GLOBALS ***
bridge = CvBridge()
img_rgb_stream = np.zeros((IMG_RES[0], IMG_RES[1], 3), np.uint8)
img_depth_stream = np.zeros((IMG_RES[0], IMG_RES[1], 1), np.uint8)
pub_raw_img = None
pub_cmpr_img = None
enable = True
rate = None

# subscribe to RGB img
def image_color_callback(data):
    global bridge, img_rgb_stream

    try:
        img_rgb_stream = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)


def image_depth_callback(data):
    global img_depth_stream, bridge

    try:
        img_depth_stream = bridge.imgmsg_to_cv2(data, "16UC1")
    except CvBridgeError as e:
        print(e)


def sign_from_segmentation():
    global img_rgb_stream, img_depth_stream

    img_orig = copy(img_rgb_stream)

    # get sign location in img
    keypoints = SegmentSign.do_hough_circle_detection(copy(img_orig), copy(img_depth_stream))
    patches = SegmentSign.get_tensor_patches(copy(img_orig), keypoints)
    if len(patches) > 0:
        return patches[0]
    else:
        return None


def publish_img_patch(img_patch):
    global bridge, pub_raw_img, pub_cmpr_img

    # use same timestamp for synchronisation
    timestamp = rospy.Time.now()

    # publish non compressed image for saving
    rawmsg = bridge.cv2_to_imgmsg(img_patch)
    rawmsg.header.stamp = timestamp
    pub_raw_img.publish(rawmsg)
    # publish compressed img for website visualization
    cmprsmsg = bridge.cv2_to_compressed_imgmsg(img_patch)
    cmprsmsg.header.stamp = timestamp
    pub_cmpr_img.publish(cmprsmsg)


def enable_callback(req):
    global enable, rate

    enable = req.data
    rospy.loginfo("({}) set enable to {}".format(rospy.get_name(), enable))
    # go in low power mode if the node is doing nothing
    if enable:
        rate = rospy.Rate(30)
    else:
        rate = rospy.Rate(5)

    return True, ""


if __name__ == "__main__":
    rospy.init_node("capture_images")

    if rospy.has_param("~remote_node"):
        REMOTE_NODE = rospy.get_param("~remote_node")
    if rospy.has_param("~save_dir"):
        SAVE_DIR = rospy.get_param("~save_dir")

    img_depth_topic = "camera/aligned_depth_to_color/image_raw"
    img_color_topic = "camera/color/image_raw"
    camera_frame = "camera_aligned_depth_to_color_frame"
    # get img stream
    rospy.Subscriber(img_color_topic, Image, image_color_callback)
    rospy.Subscriber(img_depth_topic, Image, image_depth_callback)

    if REMOTE_NODE:
        enable = False       # set this by calling the corresponding service
        rate = rospy.Rate(5)    # go in low power mode if the node is doing nothing
        # init publisher
        pub_raw_img = rospy.Publisher("~result_image", Image, queue_size=10)
        pub_cmpr_img = rospy.Publisher("~result_image/compressed", CompressedImage, queue_size=10)
        rospy.Service("~enable", SetBool, enable_callback)

        rospy.loginfo("{} is up in remote mode.".format(rospy.get_name()))
    else:
        enable = True
        rate = rospy.Rate(30)
        # init save dirs
        print("input image name: ")
        IMAGE_NAME = input()
        rospack = rospkg.RosPack()
        SAVE_DIR = "{}{}{}/".format(rospack.get_path("minibot_vision"), SAVE_DIR, IMAGE_NAME)
        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

        rospy.loginfo("{} is up in local mode. Hold r to save streamed imgs with name {} to {}".format(rospy.get_name(), IMAGE_NAME, SAVE_DIR))

    start_time = time.time()
    duration = 0.
    counter = 0
    while not rospy.is_shutdown():
        if enable:
            # from blob detector and zoomed
            img_patch = sign_from_segmentation()
            if img_patch is None:
                continue

            if REMOTE_NODE:
                publish_img_patch(img_patch)
            else:
                cv2.imshow("IMG_Color", img_patch)
                k = cv2.waitKey(1)

                if k == ord('r'):
                    duration = time.time() - start_time
                    if duration >= MAX_DURATION:
                        # reset timer
                        start_time = time.time()
                        duration = 0.

                        # save screenshot
                        img_name = "{}_{}".format(IMAGE_NAME, counter)
                        rospy.loginfo("Save img {} at {}".format(img_name, SAVE_DIR))
                        cv2.imwrite("{}{}.jpg".format(SAVE_DIR, img_name), img_patch)
                        counter += 1

        rate.sleep()

    rospy.loginfo("Node is shutting down. Closing all cv2 windows (if there are some)...")
    cv2.destroyAllWindows()
