import cv2
import src.demo as Demo


def main():
    if True:
        capture = cv2.VideoCapture('../demo1.mp4')
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 20400)
    else:
        capture = cv2.VideoCapture('../demo2.mp4')
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    
    Demo.run_demo(capture)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(cv2.__version__)
    main()