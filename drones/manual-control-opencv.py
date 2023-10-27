"""
simple example demonstrating how to control a Tello using your keyboard.

Use W, A, S, D for moving, E, Q for rotating and R, F for going up and down.
SPACE for taking pictures.
When starting the script the Tello will takeoff, pressing ESC makes it land and the script exit. 
 
Acknowledgements: script based on and inspired by: 
    - https://github.com/damiafuentes/DJITelloPy/blob/master/examples/manual-control-opencv.py
 
"""


from djitellopy import Tello
import cv2, math, time, os
from datetime import datetime
from numpy.typing import NDArray as MatLike

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

folder_path = f"images_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

tello.takeoff()

while True:
    # In reality you want to display frames in a seperate thread. Otherwise
    #  they will freeze while the drone moves.
    from typing import Optional

    img: Optional[MatLike] = frame_read.frame
    if img is None:
        print("No image")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Tello image", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord("w"):
        tello.move_forward(30)
    elif key == ord("s"):
        tello.move_back(30)
    elif key == ord("a"):
        tello.move_left(30)
    elif key == ord("d"):
        tello.move_right(30)
    elif key == ord("e"):
        tello.rotate_clockwise(30)
    elif key == ord("q"):
        tello.rotate_counter_clockwise(30)
    elif key == ord("r"):
        tello.move_up(30)
    elif key == ord("f"):
        tello.move_down(30)
    elif key == ord(" ") and img is not None:
        # check if the folder_path exists and if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(folder_path, filename), img)
        print(f"Image saved as {filename}")

tello.land()
