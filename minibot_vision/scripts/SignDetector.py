#!/usr/bin/env python3

import rospy
import std_srvs.srv

import SegmentSign
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from TmClassification import TmClassification
import cv2
from copy import copy
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from minibot_msgs.srv import set_url

# *** CONSTANTS ***
visualize = True
camera_frame = "camera_aligned_depth_to_color_frame"
IMG_RES = (480, 640)
TF_RES = (224, 224)     # tf is cropping the image

# *** GLOBALS ***
sign_classifier = TmClassification()
bridge = CvBridge()
img_rgb_stream = np.zeros((IMG_RES[0], IMG_RES[1], 3), np.uint8)
img_rgb_timestamp = rospy.Time(0, 0)
img_depth_stream = np.zeros((IMG_RES[0], IMG_RES[1], 1), np.uint8)
img_rgb = img_rgb_stream
pub_keypoint = None
pub_result_img = None

# subscribe to RGB img
def image_color_callback(data):
    global bridge, img_rgb_stream, img_rgb_timestamp

    try:
        img_rgb_stream = bridge.imgmsg_to_cv2(data, "bgr8")
        img_rgb_timestamp = rospy.Time.now()
    except CvBridgeError as e:
        print(e)


def image_depth_callback(data):
    global img_depth_stream, bridge

    try:
        img_depth_stream = bridge.imgmsg_to_cv2(data, "16UC1")
    except CvBridgeError as e:
        print(e)


def publish_results(point, radius, depth, label, precision, timestamp):
    global camera_frame

    detection_msg = Detection2D()
    # the time when the image was taken
    detection_msg.header.stamp = timestamp
    detection_msg.header.frame_id = camera_frame

    detection_msg.bbox.size_x = radius*2
    detection_msg.bbox.size_y = radius*2
    detection_msg.bbox.center.x = point[0]
    detection_msg.bbox.center.y = point[1]

    obj_with_pose = ObjectHypothesisWithPose()
    # the id might not be the same in different msgs
    #obj_with_pose.id = i
    # TODO calc x and y in img frame
    obj_with_pose.pose.pose.position.z = depth
    obj_with_pose.score = precision
    obj_with_pose.id = label

    detection_msg.results = [obj_with_pose]

    pub_keypoint.publish(detection_msg)

def detect_sign(img_rgb_stream, image_timestamp):
    global img_depth_stream, pub_result_img

    img_orig = copy(img_rgb_stream)

    # get sign location in img
    keypoints = SegmentSign.do_hough_circle_detection(copy(img_orig), copy(img_depth_stream))
    patches = SegmentSign.get_tensor_patches(copy(img_orig), keypoints)

    # cut to multiple images at keypoints
    text = []
    for i in range(len(keypoints)):
        k = keypoints[i]
        p = patches[i]
        d = k["depth"]
        center = [k["center"][1], k["center"][0]]
        r = k["radius"]

        # classify image batches
        label, precision = sign_classifier.predictImage(p)

        # publish results
        publish_results(center, r, d, label, precision, image_timestamp)
        text.append("c: {} p: {:1.3f} d:{:1.3f}".format(sign_classifier.labelOfClass(label), precision, d))

    if visualize:
        SegmentSign.visualize_patches(keypoints, patches, text, img_orig)
        # compress and publish
        cmprsmsg = bridge.cv2_to_compressed_imgmsg(img_orig)
        pub_result_img.publish(cmprsmsg)


def set_model_callback(req):
    sign_classifier.setNewModel(req.url)
    rospy.logwarn("TODO implement url error check")
    return False        # TODO implement url error check


def set_visualize_callback(req):
    global visualize

    visualize = req.data
    return True, ""


if __name__ == "__main__":
    rospy.init_node("sign_detector")

    img_color_topic = "{}camera/color/image_raw".format(rospy.get_namespace())
    img_depth_topic = "{}camera/aligned_depth_to_color/image_raw".format(rospy.get_namespace())

    rospy.Subscriber(img_color_topic, Image, image_color_callback, queue_size=1)
    rospy.Subscriber(img_depth_topic, Image, image_depth_callback, queue_size=1)
    rospy.Service('sign_detector/set_model', set_url, set_model_callback)
    rospy.Service('sign_detector/set_visualize', std_srvs.srv.SetBool, set_visualize_callback)
    pub_keypoint = rospy.Publisher('sign_detector/keypoints', Detection2D, queue_size=10)
    pub_result_img = rospy.Publisher("sign_detector/result_image/compressed", CompressedImage, queue_size=10)

    rate = rospy.Rate(30)       # currently this is impossible, but then the rate is defined by the detect_sign evaluation time
    while not rospy.is_shutdown():
        detect_sign(img_rgb_stream, img_rgb_timestamp)
        rate.sleep()
