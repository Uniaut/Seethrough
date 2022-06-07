import cv2
import numpy as np
import src.utils as utils

def process(image: cv2.Mat):
    origin = image.copy()
    image = cv2.dilate(image, np.ones((3, 3)), iterations=2)
    image = cv2.erode(image, np.ones((3, 3)), iterations=2)
    utils.imshow('TEST', image, True)
    test = cv2.subtract(image, origin)
    utils.imshow('TEST', test, True)

    return image