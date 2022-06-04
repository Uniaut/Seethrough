import cv2
from cv2 import imshow
import numpy as np

def process(keyframe: cv2.Mat, image: cv2.Mat):
    cv2.imshow('Background', keyframe)
    cv2.imshow('Now', image)
    original_image = image.copy()
    original_keyfr = keyframe.copy()


    sub = cv2.subtract(keyframe, image) + cv2.subtract(image, keyframe)

    if False:
        print('org', keyframe[0, 0, :])
        print('now', image[0, 0, :])
        print('sub', sub[0, 0, :])

    image = np.uint8(sub)
    image = cv2.erode(image, np.ones((3, 3)), iterations=2)
    cv2.imshow('erosion', image)

    blurred = image.copy()
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    retval, flooded, mask, rect = cv2.floodFill(
        blurred,
        np.zeros((blurred.shape[0] + 2, blurred.shape[1] + 2), np.uint8),
        (300, 300),
        (200, 200, 200),
        (1, 1, 1),
        (1, 1, 1)
    )
    flooded = cv2.dilate(flooded, np.ones((3, 3)), iterations=0)
    flooded = cv2.erode(flooded, np.ones((3, 3)), iterations=4)
    cv2.imshow('asdfasfdsaf', flooded)

    mask = mask[1:-1, 1:-1]
    mask = mask > 0.5
    result = original_image.copy()
    result[~mask] = result[~mask] * 0.3 + original_keyfr[~mask] * 0.7
    cv2.imshow('RESULT', result)

    if np.random.rand() < 0.03:
        new_keyfr = original_image.copy()
        new_keyfr[~mask] = original_keyfr[~mask]
        return new_keyfr
    else:
        return original_keyfr