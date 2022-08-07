import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
from copy import copy

# *** hyper params ***
IMG_RES = (480, 640)
TENSOR_RES = (224, 224)
canny = 100
VISUALIZE = False
ZOOM_THREASHOLD = 1.15      # multiplied percentage to the detected radius


# *** Globals ***
cv_bridge = CvBridge()
img_rgb_stream = np.zeros((IMG_RES[0], IMG_RES[1], 3), np.uint8)
img_depth_stream = np.zeros((IMG_RES[0], IMG_RES[1], 1), np.uint8)

def empty(d):
    pass

if VISUALIZE:
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 800, 600)
    cv2.createTrackbar("Canny1", "Parameters", canny, 255, empty)


def image_color_callback(data):
    global img_rgb_stream, cv_bridge

    try:
        img_rgb_stream = cv_bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)


def image_depth_callback(data):
    global img_depth_stream, cv_bridge

    try:
        img_depth_stream = cv_bridge.imgmsg_to_cv2(data, "16UC1")
    except CvBridgeError as e:
        print(e)


def circular_mean(p, r, arr : np.array):
    """
    returns the mean intensity in a circle described by a middle point p and radius r of a grey image.
    """
    #                   x_start         x_end       x_step  y_start         y_end       y_step
    xy = np.mgrid[int(p[0] - r) : int(p[0] + r) : 1, int(p[1] - r) : int(p[1] + r):1].reshape(2,-1).T
    sum_px_values = 0
    count_px = 0
    for x, y in xy:
        if x >= IMG_RES[1] or y >= IMG_RES[0]:
            continue
        if (x - p[0])**2 + (y - p[1])**2 < r**2:
            sum_px_values += arr[y, x]
            count_px += 1

    return sum_px_values / count_px


def do_hough_circle_detection(img_rgb, img_depth):
    global canny

    gray = img_rgb
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)      # reduce noise

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0] / 4,
                               param1=canny, param2=40,
                               minRadius=15, maxRadius=128)

    keypoints = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]

            # get depth in [m]
            d = circular_mean(center, radius * 0.4, copy(img_depth)) / 1000
            # filter if sign to close (circle detector will struggle) or to far (background)
            if d < 0.2 or d > 1.0:
                continue
            keypoints.append({"center": center, "radius": radius, "depth": d})

            # circle center
            if VISUALIZE:
                cv2.putText(img_rgb, "d:{:1.3f} r:{:1.0f}".format(d, radius), (center[0], center[1] - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
                cv2.circle(img_rgb, center, 1, (0, 100, 100), 3)
                # circle outline
                cv2.circle(img_rgb, center, radius, (255, 0, 255), 3)

    return keypoints


def crop_to_bounds(crop_bounds, max_val):
    if crop_bounds[0] < 0:
        crop_bounds[1] += 0 - crop_bounds[0]
        crop_bounds[0] = 0
    if crop_bounds[1] > max_val:
        crop_bounds[0] -= crop_bounds[1] - max_val
        crop_bounds[1] = max_val

    return crop_bounds


def get_tensor_patches(img_rgb, keypoints):
    global TENSOR_RES, ZOOM_THREASHOLD

    img_patches = []
    for k in keypoints:
        img = copy(img_rgb)
        d = k["depth"]
        center = k["center"]
        center = [center[1], center[0]]
        r = k["radius"]

        # zoom into images based on radius?
        zoom_factor = np.array(TENSOR_RES) / ((r*2 * ZOOM_THREASHOLD))
        zoomed_image = cv2.resize(img, dsize=None, fx=zoom_factor[0], fy=zoom_factor[1], interpolation=cv2.INTER_NEAREST)

        # handle border
        img_center_zoomed = (center * zoom_factor).astype(int)
        y = [img_center_zoomed[0] - TENSOR_RES[0] // 2, img_center_zoomed[0] + TENSOR_RES[0] // 2]
        y = crop_to_bounds(y, np.shape(zoomed_image)[0])
        x = [img_center_zoomed[1] - TENSOR_RES[1] // 2, img_center_zoomed[1] + TENSOR_RES[1] // 2]
        x = crop_to_bounds(x, np.shape(zoomed_image)[1])
        img_patches.append(zoomed_image[y[0]:y[1], x[0]:x[1], :])

    return img_patches


def visualize_patches(keypoints, patches, text, img_rgb):
    for i in range(len(keypoints)):
        k = keypoints[i]
        d = k["depth"]
        center = k["center"]
        center = [center[1], center[0]]
        r = k["radius"]
        patch = patches[i]

        # we need the exact idx in the non zoomed image, so we have to reacalc the boarders
        y = [center[0] - TENSOR_RES[0] // 2, center[0] + TENSOR_RES[0] // 2]
        y = crop_to_bounds(y, np.shape(img_rgb)[0])
        x = [center[1] - TENSOR_RES[1] // 2, center[1] + TENSOR_RES[1] // 2]
        x = crop_to_bounds(x, np.shape(img_rgb)[1])
        # replace the patch of the zoomed sign so that the patch that is fed to t flow can be directly seen
        img_rgb[y[0]:y[1], x[0]:x[1]] = patch
        cv2.rectangle(img_rgb, (x[0], y[0]), (x[1], y[1]), (255, 255, 255), thickness=1)
        cv2.putText(img_rgb, text[i], (x[0], y[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

    return img_rgb


if __name__=="__main__":
    rospy.init_node("hough_detection")

    VISUALIZE = True
    img_color_topic = "{}camera/color/image_raw".format(rospy.get_namespace())
    img_depth_topic = "{}camera/aligned_depth_to_color/image_raw".format(rospy.get_namespace())
    rospy.Subscriber(img_color_topic, Image, image_color_callback)
    rospy.Subscriber(img_depth_topic, Image, image_depth_callback)

    toggle_patch_visualization = True
    print("Toggle patch visualisation is {}. Press p to change.".format(toggle_patch_visualization))
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        img_processed = copy(img_rgb_stream)
        keypoints = do_hough_circle_detection(img_processed, copy(img_depth_stream))

        if toggle_patch_visualization:
            img_processed = copy(img_rgb_stream)
            patches = get_tensor_patches(copy(img_rgb_stream), keypoints)
            visualize_patches(keypoints, patches, ["d:{:1.3f}".format(k["depth"]) for k in keypoints], img_processed)

        cv2.imshow("Parameters", img_processed)
        canny = cv2.getTrackbarPos("Canny1", "Parameters")

        k = cv2.waitKey(1) & 0xFF
        if k == ord("p"):
            print("Toggle patch visualisation is {}".format(toggle_patch_visualization))
            toggle_patch_visualization = not toggle_patch_visualization


        rate.sleep()
