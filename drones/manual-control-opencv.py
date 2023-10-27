# 欲使用全手动控制请查看 manual-control-pygame.py
#
# W, A, S, D 移动， E, Q 转向，R、F上升与下降.
# 开始运行程序时Tello会自动起飞，按ESC键降落
# 并且程序会退出

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
    # 在实际开发里请在另一个线程中显示摄像头画面，否则画面会在无人机移动时静止
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
