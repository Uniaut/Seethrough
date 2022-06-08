import numpy as np
import time

import cv2
# from src.change_detection.mask import Detector
from src.segmentation.mask import Detector

import src.utils as utils
import src.postprocessing.flat as flat
import src.postprocessing.bnorm as bnorm

class Processor:
    def __init__(self, *, detector: Detector, keyframe):
        self.detector = detector
        self.keyframe = None
        self.timestamp = time.time()


    def update_keyframe(image: cv2.Mat, mask: np.ndarray):
        erased = image.copy()
        erased[mask] = keyframe[mask]
        pass


    def process(self, image: cv2.Mat):
        self.keyframe
        utils.imshow('Input Frame', image, True)
        utils.imshow('Keyframe', keyframe, True)

        mask = self.detector.get_mask(image)
        erased = image.copy()
        erased[mask] = keyframe[mask]

        # bnorm.process(erased)
        # flat.process(erased)

        now = time.time()
        if now - self.timestamp > 0.5:
            self.timestamp = now
            self.update_keyframe(erased)
        
        alpha = 1.0
        result = cv2.addWeighted(self.detector.keyframe, 0.5, image, 0.5, 0.0)
        utils.imshow('Result', result, True)


    def update_keyframe(self, image):
        self.detector.update_keyframe(image)


def run_demo(capture: cv2.VideoCapture):
    _, keyframe = capture.read()

    image_size = 0.7
    keyframe = cv2.resize(keyframe, (0, 0), fx=image_size, fy=image_size)

    detector = Detector()
    processor = Processor(detector, keyframe)

    speed = 10
    while True:
        keyframe_update = False
        keycode = cv2.waitKey(1)
        if keycode == 27:
            break
        elif keycode == ord('w'):
            keyframe_update = True
        elif keycode == ord('q'):
            speed = max(1, speed - 1)
        elif keycode == ord('e'):
            speed = min(20, speed + 1)
        

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
