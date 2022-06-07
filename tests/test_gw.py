import argparse
import cv2
import src.demo as Demo


def main():
    capture = cv2.VideoCapture('../demo.mp4')
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 20400)
    # capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    Demo.run_demo(capture)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(cv2.__version__)
    main()