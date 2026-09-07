import cv2

from asuka.utils.mediapipe_utils import MediaPipeHand


def test_MediaPipeHand(index):
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()
    hand = MediaPipeHand()
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame, _points = hand.inference(frame)
        cv2.imshow("MediaPipeHand", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


def main():
    test_MediaPipeHand(0)


if __name__ == '__main__':
    main()
