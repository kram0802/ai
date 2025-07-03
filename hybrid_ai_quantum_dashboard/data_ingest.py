import cv2

def get_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        raise RuntimeError("Failed to read frame")
