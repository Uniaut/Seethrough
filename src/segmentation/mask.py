import numpy as np

import cv2
from pixellib.instance import instance_segmentation



instance_seg = instance_segmentation(infer_speed='rapid')
instance_seg.load_model("tests\mask_rcnn_coco.h5")
target_classes = instance_seg.select_target_classes(person = True)

class Detector:
    def __init__(self, keyframe: cv2.Mat):
        self.keyframe = keyframe


    def update_keyframe(self, image: cv2.Mat):
        alpha = 0.5
        self.keyframe[:] = self.keyframe[:] * (1.0 - alpha) + image[:] * alpha


    def get_mask(self, image: cv2.Mat):
        original_shape = image.shape[:2]
        cv2.resize(image, (120, 90))
        print('image', image.shape)
        seg_mask, output = instance_seg.segmentFrame(image, target_classes)
        print('mask', seg_mask)
        np_mask = np.float32(seg_mask['masks'][:, :, 3])
        upscale_mask = cv2.resize(np_mask, original_shape)
        print('mask2', upscale_mask.shape)
        upscale_mask = upscale_mask > 0.5

        return np.transpose(upscale_mask)