import cv2
from src.change_detection.mask import Detector


def process(detector: Detector, image: cv2.Mat, keyframe_update: bool):
    if keyframe_update:
        detector.update_keyframe(image)
    mask = detector.get_mask(image, debug=True)



'''
System is composed of 2 stage:
1. Get mask and mixture with keyframe
2. Postprocess to make last stage's image into rectangular image.
'''
def run_demo(capture: cv2.VideoCapture):
    _, keyframe = capture.read()
    keyframe = cv2.resize(keyframe, (0, 0), fx=0.5, fy=0.5)
    detector = Detector(keyframe)

    while True:
        keyframe_update = False
        keycode = cv2.waitKey(1)
        if keycode == 27:
            break
        elif keycode == ord('d'):
            keyframe_update = True
        
        try:
            _, frame = capture.read()
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process(detector, frame, keyframe_update)
        except Exception as e:
            print(e)
        