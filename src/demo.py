import numpy as np
import time

import cv2
from src.change_detection.mask import Detector
# from src.segmentation.mask import Detector

import src.utils as utils
import src.postprocessing.flat as flat
import src.postprocessing.bnorm as bnorm

class Processor:
    def __init__(self, detector):
        self.detector = detector
        self.timestamp = time.time()


    def process(self, image):
        mask = self.detector.get_mask(image)
        keyframe = self.detector.keyframe
        erased = image.copy()
        alpha = 1.0
        erased[mask] = erased[mask] * (1.0 - alpha) + keyframe[mask] * alpha

        # bnorm.process(erased)
        # flat.process(erased)

        now = time.time()
        if now - self.timestamp > 0.5:
            self.timestamp = now
            self.update_keyframe(erased)
        
        result = cv2.addWeighted(self.detector.keyframe, 0.7, image, 0.3, 0.0)
        utils.imshow('Result', result, True)


    def update_keyframe(self, image):
        self.detector.update_keyframe(image)


def run_demo(capture: cv2.VideoCapture):
    _, keyframe = capture.read()

    image_size = 0.7
    keyframe = cv2.resize(keyframe, (0, 0), fx=image_size, fy=image_size)

    detector = Detector(keyframe)
    processor = Processor(detector)

    speed = 0
    while True:
        keyframe_update = False
        keycode = cv2.waitKey(1)
        if keycode == 27:
            break
        elif keycode == ord('w'):
            keyframe_update = True
        elif keycode == ord('q'):
            speed -= 1
        elif keycode == ord('e'):
            speed += 1
        

        try:
            for _ in range(speed - 1):
                capture.read()
            else:
                _, frame = capture.read()
            frame = cv2.resize(frame, (0, 0), fx=image_size, fy=image_size)

            processor.process(frame)
            if keyframe_update:
                processor.update_keyframe(frame)
        except Exception as e:
            print(e)
