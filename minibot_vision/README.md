# Minibot Vision
This package contains all the functionality related to visual tasks of the minibot.

## Startup
All of the following nodes require a running realsense camera.
Start the realsense camera with align_depth as true eg. ``roslaunch realsense2_camera rs_camera.launch align_depth:=True``.

## Segment Sign
This node detects round objects (like traffic signs) in the rgb image of the realsense camera.
The z value of the pose message is then the detected depth in the camera frame (x and y are not updated).
The detected signs are cropped to patches and zoomed so that the patch width/height matches our tensorflow requirements.
This segmentation depends on a some hyper parameters.
Since it seems to be very robust with the current configuration these are set static in the python script.
If you need to adjust them there are two different visualization modes for debugging implemented.

## Sign Detector
This node starts the SegmentSign node together with a tensorflow classification.

### Services
- **sign_detector/set_model**: With this service call a model trained and uploaded by `teachablemachine.withgoogle.com` is downloaded by the given url.
The new model overwrites the existing model and updates its label on the param server.

- **sign_detector/set_visualize**: A service to set the `visualize` flag (default: true). This flag controls the image stream on the topic `sign_detector/result_image/compressed`.
Check the Topics section for more details.

### Topics
- **sign_detector/keypoints**: A `vision_msgs.msg.Detection2D` msg will be published. 
The relevant data is stored in results as `vision_msgs.msg.ObjectHypothesisWithPose`. 
As position only the depth is set yet.
The precision of the best prediction is set as score and the corresponding label as id.
Note that the label is an integer value.
The corresponding string is stored on the parameter server under `sign_classification/class_labels/`.
As timestamp in the header the timestamp of the image used for classification is used. 
This is especially important if tensorflow is performing low to filter results that are based on outdated images.

- **sign_detector/result_image/compressed**: If the `visualize` flag is set to true a `sensor_msgs/CompressedImage` with the image and its marked and labeled detected signs is published on this topic.
The ROS tool image_view can be used for visualization: `rosrun image_view image_view image:=/<namespace>/sign_detector/result_image _image_transport:=compressed` (note that the given topic is without the `/compressed` suffix).
This is intended to be used while debugging the image classification neural network.

## Capture Images
This is a tool to capture images as training data for the sign classification network.
This will process the output of the realsense camera by SignSegmentation.
We have observed that training the network with a segmented version of the signs leads to a more robust classification.

You can run it by launching `roslaunch minibot_vision capture_imgs.launch`.
There are two arguments:
- `remote_node`: (default: true) 
  - if true the node runs on the minibot and the resulting images are published to `capture_images/result_image` as raw and compressed.
  - otherwise the node runs in local mode and the script will ask you to type the filename of the resulting images.
Then the images are visualized and can be sampled by holding `r` and are then saved to the before specified filename.
- `save_dir`: (default: "/resources/training_imgs/") only used in local mode. 
The save directory relative to the minibot_package for all images (they will be in sub-folders of their filename).

If the node is in remote mode, you need to call the service `capture_images/enable` to activate the node. 

## TODO
- [ ] Crop sign also publish x and y position in camera frame.
- [ ] SignDetector: Publish multiple keypoints (currently only one is published)