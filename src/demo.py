import numpy as np
import time

import cv2
import src.change_detection.mask as ChangeMask
import src.segmentation.mask as SegMask

import src.utils as utils
import src.postprocessing.flat as flat
import src.postprocessing.bnorm as bnorm

class Processor:
    def __init__(self, get_mask, keyframe):
        self.get_mask = get_mask
        self.keyframe = keyframe
        self.timestamp = time.time()


    def update_keyframe(self, image: cv2.Mat, mask):
        erased = image.copy()
        erased[mask] = self.keyframe[mask]

        update_rate = 0.8
        self.keyframe = cv2.addWeighted(self.keyframe, 1.0 - update_rate, erased, update_rate, 0.0)


    def process(self, image: cv2.Mat):
        utils.imshow('Input Frame', image, True)
        utils.imshow('Keyframe', self.keyframe, True)

        mask = self.get_mask(image, self.keyframe)

        # bnorm.process(erased)
        # flat.process(erased)

        now = time.time()
        if now - self.timestamp > 0.5:
            self.timestamp = now
            self.update_keyframe(image, mask)
        
        keyframe_rate = 1.0
        result = cv2.addWeighted(self.keyframe, keyframe_rate, image, 1.0 - keyframe_rate, 0.0)
        utils.imshow('Result', result, True)


def run_demo(capture: cv2.VideoCapture):
    _, keyframe = capture.read()

    image_size = 1.0
    keyframe = cv2.resize(keyframe, (0, 0), fx=image_size, fy=image_size)

    detector = ChangeMask.get_mask
    processor = Processor(detector, keyframe)

    speed = 5
    while True:
        keyframe_update = False
        keycode = cv2.waitKey(1)
        if keycode == 27:
            break
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
        except Exception as e:
            print(e)
