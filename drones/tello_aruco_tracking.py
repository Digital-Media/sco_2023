"""
A script for controlling Tello drones with ArUCo markers. The script should be extended in the course. 

Acknowledgements: script based on and inspired by: 
    - https://github.com/msoftware/tello-tracking/blob/main/tello-tracking.py
"""

import cv2
import numpy as np
from aruco_pose_estimation import PoseEstimator
from simple_pid import PID
from djitellopy import Tello


# Land if battery is below percentage
low_bat = 10


# Max velocity settings (used in limit velocity)
max_yaw_velocity = 50
max_up_down_velocity = 50
max_forward_backward_velocity = 50


def limitVelocity(velocity, max_velocity):
    return round(min(max_velocity, max(-max_velocity, velocity)))


def main():
    global low_bat, max_yaw_velocity, max_up_down_velocity, max_forward_backward_velocity

    # load the camera calibration from an openCV generated .xml file
    # Load camera calibration parameters from XML file
    calibration_data = cv2.FileStorage(
        r"./calibrations/tello_e01_calibration.xml", cv2.FILE_STORAGE_READ
    )
    k = calibration_data.getNode("Camera_Matrix").mat()
    d = calibration_data.getNode("Distortion_Coefficients").mat()
    calibration_data.release()
    estimator = PoseEstimator(k, d, debug=True)

    # Define PIDs responsible for controlling the drone
    keep_marker_at = [0, 0.0, 1]
    y_pid = PID(200, 10, 0, setpoint=keep_marker_at[1])  # up/down PID
    x_pid = PID(-100, -5, -1, setpoint=keep_marker_at[0])  # left/right PID

    # Connect to the Tello drone
    tello = Tello()
    tello.connect()

    battery = tello.get_battery()
    print("Battery: ", tello.get_battery())

    if battery < low_bat:
        print("Battery low")
        exit()

    tello.streamon()
    frame_read = tello.get_frame_read()

    tello.send_rc_control(0, 0, 0, 0)
    tello.takeoff()
    tello.move_up(30)

    winname = "Tello Aruco Tracking"
    running = True

    # main loop
    while running:
        # check battery level and stop if too low
        battery = tello.get_battery()
        if battery < low_bat:
            print("Battery low")
            running = False

        # get frames from the drone's camera
        currentFrame = frame_read.frame
        np_img = np.array(currentFrame)  # convert to numpy format (for OpenCV)
        np_img = cv2.cvtColor(
            np_img, cv2.COLOR_RGB2BGR
        )  # convert to OpenCV's internal color layout
        h, w, c = np_img.shape

        # default movements (none)
        yaw_velocity = 0
        left_right_velocity = 0
        up_down_velocity = 0
        forward_backward_velocity = 0

        # use the aruco tracker to find poses
        poses = estimator.detect(np_img)

        # if a pose has been found use the pose to move the drone
        if len(poses) > 0:
            # print(poses[0]['from_camera'])

            marker_pos = poses[0]["pos_from_camera"]  # in 3D

            up_down_velocity = limitVelocity(y_pid(marker_pos[1]), max_up_down_velocity)
            left_right_velocity = limitVelocity(
                x_pid(marker_pos[0]), max_forward_backward_velocity
            )

            print(
                "marker pos: ",
                marker_pos,
                "| controls (lr / ud): ",
                left_right_velocity,
                up_down_velocity,
            )

        # Tello remote control
        tello.send_rc_control(
            left_right_velocity,
            forward_backward_velocity,
            up_down_velocity,
            yaw_velocity,
        )

        # Show image on screen (Needs running X-Server!)
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 0, 0)
        cv2.imshow(winname, np_img)

        keyCode = cv2.waitKey(1) & 0xFF
        # Quit with Escape
        if keyCode == 27:
            break

    # Goodbye
    cv2.destroyAllWindows()
    tello.send_rc_control(0, 0, 0, 0)
    battery = tello.get_battery()
    tello.land()
    tello.streamoff()
    tello.end()


if __name__ == "__main__":
    # Execute if run as a script
    main()
