import cv2
import numpy as np
import src.utils as utils

class Detector:
    def __init__(self, keyframe: cv2.Mat):
        self.keyframe = keyframe


    def update_keyframe(self, image: cv2.Mat):
        self.keyframe = image


    def get_mask(self, image: cv2.Mat, *, debug=False):
        keyframe = self.keyframe

        diff_image = cv2.subtract(keyframe, image) + cv2.subtract(image, keyframe)

        diff_image = np.uint8(diff_image)
        diff_image = cv2.erode(diff_image, np.ones((3, 3)), iterations=2)

        blurred = diff_image.copy()
        blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
        _, flooded, mask, _ = cv2.floodFill(
            blurred,
            np.zeros((blurred.shape[0] + 2, blurred.shape[1] + 2), np.uint8),
            (300, 300),
            (200, 200, 200),
            (1, 1, 1),
            (1, 1, 1)
        )
        mask = mask[1:-1, 1:-1]
        mask = cv2.dilate(mask, np.ones((3, 3)), iterations=0)
        mask = cv2.erode(mask, np.ones((3, 3)), iterations=0)

        result = image.copy()
        mask = mask > 0.5
        result[~mask] = result[~mask] * 0.3 + keyframe[~mask] * 0.7

        utils.imshow('Input Image', image, debug)
        utils.imshow('Keyframe Image', keyframe, debug)
        utils.imshow('Erosion', diff_image, debug)
        utils.imshow('Mask', np.float32(mask), debug)
        utils.imshow('RESULT', result, debug)
        
        return mask