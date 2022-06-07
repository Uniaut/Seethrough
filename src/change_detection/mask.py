import cv2
import numpy as np

import config
import src.utils as utils

class Detector:
    def __init__(self, keyframe: cv2.Mat):
        self.keyframe = keyframe


    def update_keyframe(self, image: cv2.Mat):
        alpha = 0.3
        self.keyframe[:] = self.keyframe[:] * (1.0 - alpha) + image[:] * alpha


    def get_mask(self, image: cv2.Mat):
        keyframe = self.keyframe

        diff_image = cv2.subtract(keyframe, image) + cv2.subtract(image, keyframe)
        diff_image = cv2.GaussianBlur(diff_image, (5, 5), 0)

        diff_image = np.uint8(diff_image)
        diff_image = cv2.erode(diff_image, np.ones((3, 3)), iterations=4)
        diff_image = cv2.dilate(diff_image, np.ones((3, 3)), iterations=7)
        diff_image = cv2.erode(diff_image, np.ones((3, 3)), iterations=0)

        blurred = diff_image.copy()
        blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
        _, flooded, mask, _ = cv2.floodFill(
            blurred,
            np.zeros((blurred.shape[0] + 2, blurred.shape[1] + 2), np.uint8),
            (blurred.shape[1] // 2, 5),
            (200, 200, 200),
            (1, 1, 1),
            (1, 1, 1)
        )
        mask = mask[1:-1, 1:-1]
        mask = 1 - mask
        round_kernel = 1 - np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        mask = cv2.erode(mask, round_kernel, iterations=0)
        mask = cv2.dilate(mask, round_kernel, iterations=2)
        mask = mask > 0.5

        utils.imshow('Input Image', image, config.debug)
        utils.imshow('Keyframe Image', keyframe, config.debug)
        utils.imshow('Erosion', diff_image, config.debug)
        utils.imshow('Mask', np.float32(mask), config.debug)
        
        return mask