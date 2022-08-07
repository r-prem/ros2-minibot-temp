#!/usr/bin/env python3

# Standard imports
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
import sys
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
import distutils.util
import copy

class CropSign:
    def __init__(self, depth_topic, rgb_topic, camera_frame, visualize=False, publish=False):
        # *** params ***
        # get these from param server
        self.circularity = rospy.get_param("/crop_sign/circularity")       # 0.65
        self.thickness = rospy.get_param("/crop_sign/thickness")           # 7
        self.circle_filter = rospy.get_param("/crop_sign/circle_filter")   # 6
        self.canny1 = rospy.get_param("/crop_sign/canny1")                 # 26
        self.canny2 = rospy.get_param("/crop_sign/canny2")                 # 27

        # *** ROS topics ***
        self.camera_frame = camera_frame
        self.visualize = visualize
        self.publish = publish
        self.bridge = CvBridge()
        image_depth_sub = rospy.Subscriber(depth_topic, Image, self.image_depth_callback)
        self.img_depth_buf = np.zeros((480, 640, 1), np.uint8)

        if self.publish:
            self.pub_keypoint = rospy.Publisher('sign_keypoints', Detection2D, queue_size=10)

        if self.visualize:
            image_color_sub = rospy.Subscriber(rgb_topic, Image, self.image_color_callback)

            cv2.namedWindow("Parameters")
            cv2.resizeWindow("Parameters", 800, 600)
            cv2.createTrackbar("Circularity", "Parameters", int(self.circularity * 100), 100, self.empty)
            cv2.createTrackbar("Thickness", "Parameters", self.thickness, 30, self.empty)
            cv2.createTrackbar("CircleFilter", "Parameters", self.circle_filter, 30, self.empty)
            cv2.createTrackbar("Canny1", "Parameters", self.canny1, 255, self.empty)
            cv2.createTrackbar("Canny2", "Parameters", self.canny2, 255, self.empty)

            self.img_contours = np.zeros((480, 640, 3), np.uint8)
            self.img_rgb = np.zeros((480, 640, 3), np.uint8)
            self.ros_img_rgb = np.zeros((480, 640, 3), np.uint8)
            self.img_depth = np.zeros((480, 640, 1), np.uint8)
            cv2.namedWindow("RGB")
            cv2.namedWindow("Depth")

    def spin(self):
        rate = rospy.Rate(10)
        try:
            if visualize:
                while not rospy.is_shutdown():
                    # *** update parameters ***
                    self.circularity = cv2.getTrackbarPos("Circularity", "Parameters") / 100
                    self.thickness = cv2.getTrackbarPos("Thickness", "Parameters")
                    self.circle_filter = cv2.getTrackbarPos("CircleFilter", "Parameters")
                    self.canny1 = cv2.getTrackbarPos("Canny1", "Parameters")
                    self.canny2 = cv2.getTrackbarPos("Canny2", "Parameters")

                    # *** show image stream to update parameters ***
                    cv2.imshow('Parameters', self.img_contours)
                    cv2.imshow('RGB', self.img_rgb)
                    cv2.imshow('Depth', self.img_depth)
                    cv2.waitKey(1)
                    rate.sleep()
            else:
                rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")

        cv2.destroyAllWindows()


    def empty(self, d):
        pass

    def blob_detector(self, im):
        # Set up the detector with default parameters.
        # Setup SimpleBlobDetector parameters.
        # find a detailed description in the official docu: https://docs.opencv.org/4.x/d0/d7a/classcv_1_1SimpleBlobDetector.html
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        # params.minThreshold = 20
        # params.thresholdStep = 5
        # params.maxThreshold = 150

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 10000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = self.circularity
        params.maxCircularity = 1.0

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(im)

        return keypoints

    def depth_edge_detector(self, img : np.array, canny1, canny2):
        """
        This function does some preprocessing to a discretized grey img [0,255] to detect (round) edges.
        :param canny1 First param of canny edge detector
        :param canny2 Second param of canny edge detector
        """
        # circular kernel makes noisy blobs more round
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.circle_filter, self.circle_filter))
        img_filtered = cv2.dilate(img, kernel, iterations=2)
        img_filtered = cv2.GaussianBlur(img_filtered, (9, 9), 0)
        img_canny = cv2.Canny(img_filtered, canny1, canny2)

        return img_canny, img_filtered

    def discretize_depth_img(self, img : np.array, max_range : int =1000):
        """
        This function converts a depth image with integer range [mm] to values (near to wide) between [0, 255].
        :param max_range: The max range in the converted image. Values above will be clipped to 255.
        :return The converted img.
        """
        img_d = np.clip(img, 0, max_range)     # 1m is the max range we are supporting
        img_d = np.array((img_d / max_range) * 255, dtype=np.uint8)     # convert to values [0, 255]

        return img_d

    def image_color_callback(self, data):
        try:
            self.ros_img_rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def image_depth_callback(self, data):
        try:
            self.img_depth_buf = self.bridge.imgmsg_to_cv2(data, "16UC1")
            # only run the automated detection if this is supposed to run as standalone node
            if self.publish or self.visualize:
                self.detect_signs_depth_img()
        except CvBridgeError as e:
            print(e)

    def circular_mean(self, p, r, arr : np.array):
        """
        returns the mean intensity in a circle described by a middle point p and radius r of a grey image.
        """
        #                   x_start         x_end       x_step  y_start         y_end       y_step
        xy = np.mgrid[int(p[0] - r) : int(p[0] + r) : 1, int(p[1] - r) : int(p[1] + r):1].reshape(2,-1).T
        sum_px_values = 0
        count_px = 0
        for x, y in xy:
            if (x - p[0])**2 + (y - p[1])**2 < r**2:
                sum_px_values += arr[y, x]
                count_px += 1

        return sum_px_values / count_px

    def detect_signs_depth_img(self):
        depth_img_raw = copy.copy(self.img_depth_buf)       # to ensure that there is no raise condition
        depth_img_discret = self.discretize_depth_img(depth_img_raw)
        img_edge, img_depth_filtered = self.depth_edge_detector(depth_img_discret, self.canny1, self.canny2)

        #img_contour = self.contour_detector(img_edge)
        img_contour = img_edge

        kernel = np.ones((self.thickness, self.thickness))
        img_contour = cv2.dilate(img_contour, kernel, iterations=1)
        keypoints = self.blob_detector(img_contour)  # the blob detector is detecting black blobs in white backgrounds

        # filter depth shadows out
        new_keypoints = []
        for p in keypoints:
            if depth_img_raw[int(p.pt[1]), int(p.pt[0])] > 0:       # TODO do this by taking the most common value of the blob
                new_keypoints.append(p)

        if self.visualize:
            # Draw detected blobs as red circles in contour image.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            img_with_keypoints = cv2.drawKeypoints(img_contour, keypoints, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # update images
            self.img_contours = img_with_keypoints
            self.img_depth = img_depth_filtered
            self.img_rgb = self.ros_img_rgb.copy()      # copy the image since the asynchron callback could override any changes

        # publish keypoints
        depths = []
        for i, (p) in enumerate(new_keypoints):
            # depth in [m]
            depth = self.circular_mean(p.pt, (p.size/2) * 0.1, depth_img_raw) / 1000      # take the depth value from the original image (10% of radius)
            depths.append(depth)

            if self.publish:
                detection_msg = Detection2D()
                detection_msg.header.stamp = rospy.Time.now()
                detection_msg.header.frame_id = self.camera_frame       # TODO check if this is using tf_prefix

                detection_msg.bbox.size_x = p.size
                detection_msg.bbox.size_y = p.size
                detection_msg.bbox.center.x = int(p.pt[0])
                detection_msg.bbox.center.y = int(p.pt[1])

                obj_with_pose = ObjectHypothesisWithPose()
                # the id might not be the same in different msgs
                obj_with_pose.id = i
                # TODO calc x and y in img frame
                obj_with_pose.pose.pose.position.z = depth
                detection_msg.results = [obj_with_pose]

                self.pub_keypoint.publish(detection_msg)

            # visualize filtered keypoints on rgb image
            if self.visualize:
                cv2.circle(self.img_rgb,(int(p.pt[0]),int(p.pt[1])), int(p.size/2), (0, 0, 0), thickness=2)
                cv2.putText(self.img_rgb, "d:{}".format(depth), (int(p.pt[0]), int(p.pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)

        return new_keypoints, depths

if __name__ == "__main__":
    # parse args
    visualize = True
    for i, arg in enumerate(sys.argv):
        if i == 1:
            visualize = bool(distutils.util.strtobool(arg))

    rospy.init_node("crop_sign")

    img_color_topic = "{}camera/color/image_raw".format(rospy.get_namespace())
    img_depth_topic = "{}camera/aligned_depth_to_color/image_raw".format(rospy.get_namespace())
    camera_frame = "camera_aligned_depth_to_color_frame"

    sign_detector = CropSign(img_depth_topic, img_color_topic, camera_frame, visualize=visualize)
    rospy.loginfo("{}: SignDetector is up with visualize={}. Spinning ...".format(rospy.get_name(), visualize))
    sign_detector.spin()

