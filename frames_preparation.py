import cv2
import numpy as np


def prepare(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resize = cv2.resize(gray, (284, 360))
    gray_resize_norm = gray_resize / 255.0

    gray_resize_norm = np.expand_dims(gray_resize_norm, axis=2)

    return gray_resize_norm


def prepare_rgb(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resize = cv2.resize(frame, (284, 360))
    frame_resize_norm = frame_resize / 255.0

    return frame_resize_norm