import cv2
import numpy as np

def merge_images(original, encoded):
    merged = cv2.addWeighted(original, 0.5, encoded, 0.5, 0)
    return merged