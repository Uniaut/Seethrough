import cv2


def imshow(title: str, image: cv2.Mat, debug: bool):
    if debug:
        cv2.imshow(title, image)