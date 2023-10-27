"""An object detector using Google's MediaPipe framework 

https://developers.google.com/mediapipe/solutions/vision/object_detector#models
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return image


class ObjectDetector:
    def __init__(
        self,
        model_path="./models/efficientdet.tflite",
        detection_threshold=0.5,
        debug=True,
    ):
        self.debug = debug
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, score_threshold=detection_threshold
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, frame):
        # Convert the OpenCV image to RGB format
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the RGB image to MediaPipe image format
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect objects in the input image.
        detection_result = self.detector.detect(image)

        if self.debug:
            # Process the detection result. In this case, visualize it.
            frame = visualize(frame, detection_result)

        return detection_result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--ModelPath",
        required=False,
        help="Path to model (tflite file)",
        default="./models/efficientdet.tflite",
    )
    ap.add_argument(
        "-i",
        "--Image",
        required=False,
        help="Path to image file",
        default="./images/image_20231016_145650.jpg"
        # default="./images/image_20230915_175417.jpg",
    )
    ap.add_argument(
        "-t",
        "--Threshold",
        required=False,
        help="Detection threshold",
        default=0.2,
    )

    args = vars(ap.parse_args())

    detector = ObjectDetector(args["ModelPath"], args["Threshold"], debug=True)

    image_path = args["Image"]
    image = None
    if image_path is not None:
        image = cv2.imread(image_path)
        video = None
    else:
        video = cv2.VideoCapture(0)

    while True:
        if video is not None:
            ret, frame = video.read()
            if not ret:
                break
        elif image is not None:
            frame = image.copy()
        else:
            print("No Video and no Image given")
            break

        poses = detector.detect(frame)

        cv2.imshow("Detected Objects", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    if video:
        video.release()
    cv2.destroyAllWindows()
