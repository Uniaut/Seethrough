import csv
from typing import Literal
import numpy as np
import time

import cv2
import tqdm

import config
import src.change_detection.mask as ChangeMask
import src.segmentation.mask as SegMask
import src.utils as utils

class Processor:
    def __init__(self, get_mask, keyframe):
        self.get_mask = get_mask
        self.keyframe = keyframe
        self.timestamp = time.time()


    def update_keyframe(self, image: cv2.Mat, mask):
        erased = image.copy()
        erased[mask] = self.keyframe[mask]

        update_rate = 0.1
        self.keyframe = cv2.addWeighted(self.keyframe, 1.0 - update_rate, erased, update_rate, 0.0)


    def process(self, image: cv2.Mat):
        result_mode = config.visual_mode
        utils.imshow('Input Frame', image, result_mode)
        utils.imshow('Keyframe', self.keyframe, result_mode)

        mask = self.get_mask(image, self.keyframe)

        now = time.time()
        if now - self.timestamp > 0.05:
            self.timestamp = now
            self.update_keyframe(image, mask)

        keyframe_rate = 0.75
        result = cv2.addWeighted(self.keyframe, keyframe_rate, image, 1.0 - keyframe_rate, 0.0)
        utils.imshow('Result', result, result_mode)


def run_demo(capture: cv2.VideoCapture):
    _, keyframe = capture.read()

    image_size = 1.0
    keyframe = cv2.resize(keyframe, (0, 0), fx=image_size, fy=image_size)

    mode = config.mask_mode
    if mode == 1:
        detector = SegMask.get_mask
    else:
        detector = ChangeMask.get_mask
    processor = Processor(detector, keyframe)

    speed = 1
    while True:
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


def speed_test(capture: cv2.VideoCapture):
    _, keyframe = capture.read()

    mask_mode = config.mask_mode
    if mask_mode == 0:
        detector = ChangeMask.get_mask
    if mask_mode == 1:
        detector = SegMask.get_mask
    
    processor = Processor(detector, keyframe)

    timestamp = time.time()
    inference_time = []
    for _ in tqdm.tqdm(range(500)):
        try:
            _, frame = capture.read()
            processor.process(frame)
        except Exception as e:
            print(e)

        try:
            dT = time.time() - timestamp
            inference_time.append(dT)
            timestamp = time.time()
        except:
            pass

    inference_time.sort()
    inference_time = np.float32(inference_time)
    print('Avg:', np.average(inference_time))
    print('1%:', np.average(inference_time[-100:]))
    with open('test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for dt in inference_time.tolist():
            writer.writerow((dt,))
