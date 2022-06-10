import cv2
import src.demo as Demo
import config


def main():
    mode = config.video_option
    if mode == 1:
        capture = cv2.VideoCapture('../demo1.mp4')
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 20400)
    elif mode == 2:
        capture = cv2.VideoCapture('../demo2.mp4')
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    else:
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if config.run_mode == 'demo':
        Demo.run_demo(capture)
    else:
        Demo.speed_test(capture)

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(cv2.__version__)
    main()