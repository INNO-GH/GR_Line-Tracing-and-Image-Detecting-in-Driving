from jetracer.nvidia_racecar import NvidiaRacecar
import os
import cv2
from datetime import datetime
import pygame
from IPython.display import display, Image
import ipywidgets as widgets
import threading
from jetcam.utils import bgr8_to_jpeg
from jetcam.csi_camera import CSICamera
from time import sleep

# ====================

# 카메라에 대한 레코딩을 계속 관장하는 스레드 함수
def view():
    frame = camera.read()
    frame_index = 0

    while True:
        if video:
            print("Record")
            frame = camera.read()
            if frame is not None:
                image_filename = os.path.join("image", f"frame_{frame_index:09d}.jpg")
                cv2.imwrite(image_filename, frame[:,:,::-1])  # OpenCV는 BGR 형식을 사용하므로 RGB로 변환하여 저장
                print(f"Image saved as {image_filename}")
                frame_index += 1
                sleep(1)
            
# ====================

# 1. 젯레이서 가져오기
car = NvidiaRacecar()
car.steering_offset = 0.08
car.throttle_gain = 0.5

# 2. 조이스틱 가져오기
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()
running = True
throttle_range = (-0.4, 0.4)

# 3. 스레드 하나 열어서, 조이스틱 버튼 누르면 젯레이서 카메라 저장
video = False
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)
thread = threading.Thread(target=view, args=())
thread.start()

# 4. 젯레이서 조이스틱으로 조작하기
while running:
    pygame.event.pump()

    throttle = -joystick.get_axis(1)
    throttle = max(throttle_range[0], min(throttle_range[1], throttle))
    steering = joystick.get_axis(2)

    #print(throttle, steering)
    car.throttle = throttle
    car.steering = steering
    if joystick.get_button(10): # select button
            video = True
    if joystick.get_button(11): # start button
            video = False
    if joystick.get_button(12): # home button
        running = False
        camera.release()
