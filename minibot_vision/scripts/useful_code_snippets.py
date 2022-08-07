import cv2
import numpy as np

def to_HSV(im):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H = im_hsv[:, :, 0]
    S = im_hsv[:, :, 1]
    V = im_hsv[:, :, 2]

    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def background_blur(img, depth_img):
    # get depth mask
    depth_mask = np.where((depth_img > 1000) | (depth_img <= 0), 0., 1.)  # 0 at every pos depth > 1m
    # make coarser edges
    kernel = np.ones((7, 7))
    depth_mask = cv2.dilate(depth_mask, kernel, iterations=1)

    im_blurred = cv2.GaussianBlur(img, (21, 21), 15)
    im = np.where(depth_mask == 0, im_blurred, img)

def contour_detector(self, im):
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw contours
    contours_img = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    contours_img = cv2.drawContours(contours_img, contours, -1, (255, 255, 255), thickness=self.thickness, hierarchy=hierarchy,
                                    maxLevel=self.max_depth)

    return contours_img

