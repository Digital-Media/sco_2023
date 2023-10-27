"""
A python script/class for tracking Aruco markers and estimating the camera pose with OpenCV.

You can create markers here: 
    https://chev.me/arucogen/

Acknowledgements: script based on and inspired by: 
    https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/tree/main
    https://github.com/tizianofiorenzani/how_do_drones_work/blob/master/opencv/aruco_pose_estimation.py
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    
"""


import numpy as np
import cv2
import argparse
import math


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class PoseEstimator:
    def __init__(
        self,
        matrix_coefficients,
        distortion_coefficients,
        marker_size=0.10,  # meter
        marker_ids=0,
        aruco_dict_type=cv2.aruco.DICT_4X4_50,
        debug=True,
    ):
        self.aruco_dict_type = aruco_dict_type
        self.marker_size = marker_size
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients
        self.debug = debug
        self.marker_ids = (
            [marker_ids] if not isinstance(marker_ids, list) else marker_ids
        )

        try:
            self.aruco_dict = cv2.aruco.Dictionary_get(self.aruco_dict_type)
            self.parameters = cv2.aruco.DetectorParameters_create()
        except:
            # try a newer version of opencv (see https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)
            self.parameters = cv2.aruco.DetectorParameters()

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )

        poses = []
        if len(corners) > 0:
            for i in range(0, len(ids)):
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i],
                    self.marker_size,
                    self.matrix_coefficients,
                    self.distortion_coefficients,
                )
                rmat = np.matrix(cv2.Rodrigues(rvec)[0])

                # The markers position and attitude in the camera frame. Note that the camera's y axis points down and the z away from the camera
                marker_pos = tvec[0, 0, :]
                marker_euler = rotationMatrixToEulerAngles(rmat)

                # the camera's position and attitude in the marker' coordinate system
                camera_pos = -rmat.T * np.matrix(tvec[0, 0, :]).T
                camera_euler = rotationMatrixToEulerAngles(rmat.T)

                if self.debug:
                    fontScale = 0.5
                    cv2.drawFrameAxes(
                        frame,
                        self.matrix_coefficients,
                        self.distortion_coefficients,
                        rvec,
                        tvec,
                        self.marker_size / 2,
                    )
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    # print the marker's position and attitude in the camera frame
                    str_position = "MARKER Position x=%4.1f  y=%4.1f  z=%4.1f" % (
                        marker_pos[0],
                        marker_pos[1],
                        marker_pos[2],
                    )
                    cv2.putText(
                        frame,
                        str_position,
                        (0, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    str_attitude = "MARKER Attitude r=%4.1f  p=%4.1f  y=%4.1f" % (
                        math.degrees(marker_euler[0]),
                        math.degrees(marker_euler[1]),
                        math.degrees(marker_euler[2]),
                    )
                    cv2.putText(
                        frame,
                        str_attitude,
                        (0, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # print the camera's position and attitude in the marker' coordinate system
                    str_position = "CAMERA Position x=%4.1f  y=%4.1f  z=%4.1f" % (
                        camera_pos[0],
                        camera_pos[1],
                        camera_pos[2],
                    )
                    cv2.putText(
                        frame,
                        str_position,
                        (0, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    str_attitude = "CAMERA Attitude r=%4.1f  p=%4.1f  y=%4.1f" % (
                        math.degrees(camera_euler[0]),
                        math.degrees(camera_euler[1]),
                        math.degrees(camera_euler[2]),
                    )
                    cv2.putText(
                        frame,
                        str_attitude,
                        (0, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if ids[i] in self.marker_ids:
                    poses.append(
                        {
                            "rvec": rvec,
                            "tvec": tvec,
                            "euler_from_camera": marker_euler,
                            "pos_from_camera": marker_pos,
                        }
                    )

        return poses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--Calibration",
        required=False,
        help="Path to calibration (xml file)",
        default="./calibrations/tello_e01_calibration.xml",
    )
    ap.add_argument(
        "-i",
        "--Image",
        required=False,
        help="Path to image file",
        default="./images/aruco_4x4_0_z50cm_y10_image.jpg",
    )
    ap.add_argument(
        "-s",
        "--MarkerSize",
        required=False,
        help="Marker size in meters",
        default=0.10,
        type=float,
    )
    ap.add_argument(
        "-m", "--MarkerId", required=False, help="Marker ID", default=0, type=int
    )
    args = vars(ap.parse_args())

    aruco_dict_type = cv2.aruco.DICT_4X4_50  # cv2.aruco.DICT_6X6_250
    calibration_file_path = args["Calibration"]
    image_path = args["Image"]
    marker_size = args["MarkerSize"]
    marker_id = args["MarkerId"]

    # load the camera calibration from an openCV generated .xml file
    # Load camera calibration parameters from XML file
    calibration_data = cv2.FileStorage(calibration_file_path, cv2.FILE_STORAGE_READ)
    k = calibration_data.getNode("Camera_Matrix").mat()
    d = calibration_data.getNode("Distortion_Coefficients").mat()
    calibration_data.release()

    estimator = PoseEstimator(k, d, marker_size, marker_id, aruco_dict_type, debug=True)

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

        poses = estimator.detect(frame)

        cv2.imshow("Estimated Pose", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if video:
        video.release()
    cv2.destroyAllWindows()


# create markers here: https://chev.me/arucogen/
