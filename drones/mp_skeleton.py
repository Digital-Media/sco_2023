"""A skeleton detector using Google's MediaPipe framework 

https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import argparse


def draw_landmarks_on_image(rgb_image, detection_result, use_segmentation=False):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = rgb_image  # copy by reference (this overwrites rgb_image)

    mask = np.ones(rgb_image.shape[:2], dtype=np.uint8)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        if use_segmentation:
            mask[detection_result.segmentation_masks[0].numpy_view() > 0.1] += 1

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

    if use_segmentation:
        if np.max(mask) > 1:
            mask -= 1
        rgb_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # mask non-skeleton areas with black
        annotated_image[rgb_mask < 1] = 0

    return annotated_image


class ObjectDetector:
    def __init__(
        self,
        model_path="./models/pose_landmarker_lite.task",
        compute_segmentation=True,
        debug=True,
    ):
        self.debug = debug
        self.compute_segmentation = compute_segmentation
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options, output_segmentation_masks=compute_segmentation
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        # Convert the OpenCV image to RGB format
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the RGB image to MediaPipe image format
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect objects in the input image.
        detection_result = self.detector.detect(image)

        if self.debug:
            # Process the detection result. In this case, visualize it.
            frame = draw_landmarks_on_image(
                frame, detection_result, use_segmentation=self.compute_segmentation
            )

        return detection_result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--ModelPath",
        required=False,
        help="Path to model (tflite file)",
        default="./models/pose_landmarker_lite.task",
    )
    ap.add_argument(
        "-i",
        "--Image",
        required=False,
        help="Path to image file",
        # default="./images/image_20230915_175417.jpg",
        default="./images/image_20231016_145650.jpg",
    )
    ap.add_argument(
        "-t",
        "--Threshold",
        required=False,
        help="Detection threshold",
        default=0.5,
    )

    args = vars(ap.parse_args())

    detector = ObjectDetector(args["ModelPath"], args["Threshold"], debug=True)

    image_path = args["Image"]
    if image_path is not None:
        image = cv2.imread(image_path)
        video = None
    else:
        video = cv2.VideoCapture(0)

    while True:
        if video:
            ret, frame = video.read()
            if not ret:
                break
        else:
            frame = image.copy()

        poses = detector.detect(frame)

        cv2.imshow("Detected Objects", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    if video:
        video.release()
    cv2.destroyAllWindows()


# create markers here: https://chev.me/arucogen/
