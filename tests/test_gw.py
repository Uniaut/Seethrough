import cv2
import src.demo as Demo

def main():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    Demo.run_demo(capture)

if __name__ == '__main__':
    print(cv2.__version__)
    main()