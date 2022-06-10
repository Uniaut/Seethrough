import numpy as np

import cv2
from pixellib.torchbackend.instance import instanceSegmentation

import config
import src.utils as utils


instance_seg = instanceSegmentation()
instance_seg.load_model("..\pointrend_resnet50.pkl", detection_speed='fast')
target_classes = instance_seg.select_target_classes(person=True)


def get_mask(image: cv2.Mat, *_):
    original_shape = image.shape[:2]
    mask_shape = (original_shape[1], original_shape[0])
    image = cv2.resize(image, (90, 60))

    utils.imshow('input_image', image, config.debug)
    
    seg_mask, output = instance_seg.segmentFrame(image, target_classes)
    utils.imshow('outputputput', output, config.debug)
    # print('mask', seg_mask)
    np_mask = np.float32(seg_mask['masks'])
    if not np_mask.size:
        return np.zeros(original_shape, np.bool8)
    np_mask = np.sum(np_mask, axis=2)

    np_mask = cv2.dilate(np_mask, np.ones((3, 3)), iterations=1)
    utils.imshow('MASK_', np.float32(np_mask > 0.5), config.debug)
    
    upscale_mask = cv2.resize(np_mask, mask_shape)
    upscale_mask = upscale_mask > 0.5
    # print(upscale_mask)

    utils.imshow('MASK', np.float32(upscale_mask), config.debug)
    return upscale_mask