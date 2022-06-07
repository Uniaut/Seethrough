import cv2
import numpy as np
import src.utils as utils
import config

def process(image: cv2.Mat):
    origin = image.copy()
    image = cv2.dilate(image, np.ones((3, 3)), iterations=3)
    image = cv2.erode(image, np.ones((3, 3)), iterations=6)
    image = cv2.dilate(image, np.ones((3, 3)), iterations=3)
    utils.imshow('bnorm.tophat', image, config.debug)
    on_board = cv2.subtract(image, origin)
    result = cv2.subtract(image, on_board)
    utils.imshow('bnorm.result', result, config.debug)
    return result