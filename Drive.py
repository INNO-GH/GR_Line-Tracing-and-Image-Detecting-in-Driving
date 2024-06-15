from cnn.center_dataset import TEST_TRANSFORMS
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
import torch
import torchvision
import PIL.Image
import copy
import numpy as np
from ultralytics import YOLO
import re

# ====================

# Function Get Model
def get_model():
    model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return model

# Function PreProcess
def preprocess(capture: PIL.Image):
    device = torch.device('cuda')    
    capture = TEST_TRANSFORMS(capture).to(device)
    return capture[None, ...]

# Function Mode Select
def mode_select(text, lock_entrance_number, no_detection_lock, crosswalk_after_mode):
    global Mode, Lock, LockStart
    if(Lock <= lock_entrance_number):    
        if(text[8]==']'):
            Mode = 1
            Lock = no_detection_lock
        elif(text[8]=='0'):
            Mode = 4
            Lock = LockStart
        elif(text[8]=='1'):
            Mode = crosswalk_after_mode
            Lock = LockStart
        elif(text[8]=='2'):
            Mode = 2
            Lock = LockStart
        elif(text[8]=='3'):
            Mode = 3
            Lock = LockStart
        elif(text[8]=='4'):
            Mode = 1
            Lock = LockStart
    Lock = max(0, Lock-1)

# ====================

# 1. Set JoyStick
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()
Mode = 0
Lock = 0
LockStart = 16

# 2. Set JetRacer
car = NvidiaRacecar()
car.steering_offset = 0.08
car.throttle_gain = 0.5
ThrottleNormal = 0.33

# 3. Set Camera 
camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)

# 4. Set Model
    ##### Straight #####
model = get_model()
model.load_state_dict(torch.load('Follow_Model.pth'))
device = torch.device('cuda')
model = model.to(device)
    ##### Left #####
model_left = get_model()
model_left.load_state_dict(torch.load('Follow_Model_Left.pth'))
device = torch.device('cuda')
model_left = model_left.to(device)
    ##### Right #####
model_right = get_model()
model_right.load_state_dict(torch.load('Follow_Model_Right.pth'))
device = torch.device('cuda')
model_right = model_right.to(device)
    ##### Signal #####
model_signal = YOLO('Signal_Model.pt', task='detect')
device = torch.device('cuda')
model_signal = model_signal.to(device)

# ====================

# LineTracing/SignalDetecting (JoyStick -> Camera -> Model -> JetRacer)
while True:

    pygame.event.pump()

    ##### Base #####
    if Mode==0:
        
        # JoyStick
        if joystick.get_button(10):
            Mode = 1
            Lock = 0
            continue

        # JetRacer
        car.throttle = 0
        car.steering = 0

    ##### Straight #####
    elif Mode==1:
    
        # JoyStick
        if joystick.get_button(11):
            Mode = 0
            Lock = 0
            continue

        # Camera
        frame = camera.read()
        if frame is not None:
            capture_filename = os.path.join("capture", f"capture.jpg")
            cv2.imwrite(capture_filename, frame[:,:,::-1])

        # Model
        capture_filename_fmt = 'capture/capture.jpg'
        capture_ori = PIL.Image.open(capture_filename_fmt)
        width = capture_ori.width
        height = capture_ori.height
        with torch.no_grad():
            capture = preprocess(capture_ori)
            output = model(capture).detach().cpu().numpy()
        x, y = output[0]
        x = (x / 2 + 0.5) * width
        y = (y / 2 + 0.5) * height
        result = model_signal.predict(source='capture', save=True)   
        text = str(result[0].__dict__['boxes'].cls)
        mode_select(text, 0, 0, 5)

        # JetRacer
        car.steering = np.tanh(0.1*(x-315))
        car.throttle = 0
        sleep(0.1)
        car.throttle = 0.1
        car.throttle = 0.2
        car.throttle = ThrottleNormal
        sleep(0.2)
        car.throttle = 0
        sleep(0.1)

    ##### Left #####
    elif Mode==2:
    
        # JoyStick
        if joystick.get_button(11):
            Mode = 0
            Lock = 0
            continue

        # Camera
        frame = camera.read()
        if frame is not None:
            capture_filename = os.path.join("capture", f"capture.jpg")
            cv2.imwrite(capture_filename, frame[:,:,::-1])

        # Model
        capture_filename_fmt = 'capture/capture.jpg'
        capture_ori = PIL.Image.open(capture_filename_fmt)
        width = capture_ori.width
        height = capture_ori.height
        with torch.no_grad():
            capture = preprocess(capture_ori)
            output = model_left(capture).detach().cpu().numpy()
        x, y = output[0]
        x = (x / 2 + 0.5) * width
        y = (y / 2 + 0.5) * height
        result = model_signal.predict(source='capture', save=True)   
        text = str(result[0].__dict__['boxes'].cls)
        mode_select(text, 0, 0, 5)

        # JetRacer
        car.steering = np.tanh(0.1*(x-315))
        car.throttle = 0
        sleep(0.1)
        car.throttle = 0.1
        car.throttle = 0.2
        car.throttle = ThrottleNormal
        sleep(0.2)
        car.throttle = 0
        sleep(0.1)

    ##### Right #####
    elif Mode==3:
    
        # JoyStick
        if joystick.get_button(11):
            Mode = 0
            Lock = 0
            continue

        # Camera
        frame = camera.read()
        if frame is not None:
            capture_filename = os.path.join("capture", f"capture.jpg")
            cv2.imwrite(capture_filename, frame[:,:,::-1])

        # Model
        capture_filename_fmt = 'capture/capture.jpg'
        capture_ori = PIL.Image.open(capture_filename_fmt)
        width = capture_ori.width
        height = capture_ori.height
        with torch.no_grad():
            capture = preprocess(capture_ori)
            output = model_right(capture).detach().cpu().numpy()
        x, y = output[0]
        x = (x / 2 + 0.5) * width
        y = (y / 2 + 0.5) * height
        result = model_signal.predict(source='capture', save=True)   
        text = str(result[0].__dict__['boxes'].cls)
        mode_select(text, 0, 0, 5)

        # JetRacer
        car.steering = np.tanh(0.1*(x-315))
        car.throttle = 0
        sleep(0.1)
        car.throttle = 0.1
        car.throttle = 0.2
        car.throttle = ThrottleNormal
        sleep(0.2)
        car.throttle = 0
        sleep(0.1)

    ##### Bus #####
    elif Mode==4:
    
        # JoyStick
        if joystick.get_button(11):
            Mode = 0
            Lock = 0
            continue

        # Camera
        frame = camera.read()
        if frame is not None:
            capture_filename = os.path.join("capture", f"capture.jpg")
            cv2.imwrite(capture_filename, frame[:,:,::-1])

        # Model
        capture_filename_fmt = 'capture/capture.jpg'
        capture_ori = PIL.Image.open(capture_filename_fmt)
        width = capture_ori.width
        height = capture_ori.height
        with torch.no_grad():
            capture = preprocess(capture_ori)
            output = model(capture).detach().cpu().numpy()
        x, y = output[0]
        x = (x / 2 + 0.5) * width
        y = (y / 2 + 0.5) * height
        result = model_signal.predict(source='capture', save=True)   
        text = str(result[0].__dict__['boxes'].cls)
        mode_select(text, 0, 0, 5)

        # JetRacer
        car.steering = np.tanh(0.1*(x-315))
        car.throttle = 0
        sleep(0.2)
        car.throttle = 0.1
        car.throttle = 0.2
        car.throttle = ThrottleNormal
        sleep(0.2)
        car.throttle = 0
        sleep(0.2)

    ##### Crosswalk #####
    elif Mode==5:
    
        # JoyStick
        if joystick.get_button(11):
            Mode = 0
            Lock = 0
            continue

        # Camera
        frame = camera.read()
        if frame is not None:
            capture_filename = os.path.join("capture", f"capture.jpg")
            cv2.imwrite(capture_filename, frame[:,:,::-1])

        # Model
        capture_filename_fmt = 'capture/capture.jpg'
        capture_ori = PIL.Image.open(capture_filename_fmt)
        width = capture_ori.width
        height = capture_ori.height
        with torch.no_grad():
            capture = preprocess(capture_ori)
            output = model(capture).detach().cpu().numpy()
        x, y = output[0]
        x = (x / 2 + 0.5) * width
        y = (y / 2 + 0.5) * height
        result = model_signal.predict(source='capture', save=True)   
        text = str(result[0].__dict__['boxes'].cls)
        mode_select(text, 1000, LockStart, 1)

        # JetRacer
        car.steering = np.tanh(0.1*(x-315))
        car.throttle = 0
        sleep(5)
        car.throttle = 0.1
        car.throttle = 0.2
        car.throttle = ThrottleNormal
        sleep(0.2)
        car.throttle = 0
        sleep(0.1)