import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2
import os, sys

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3,320)
cam.set(4,240)
ins = instanceSegmentation()
ins.load_model("..\src\pointrend_resnet50.pkl")
target_classes = ins.select_target_classes(person = True)

while True:
    ret, frame = cam.read()
    cv2.imshow("frame", frame)    
    result, output = ins.segmentFrame(frame)
    cv2.imshow("result", output)        
    if cv2.waitKey(25) == 27:
        break