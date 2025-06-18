# postprocess.py
import cv2
import numpy as np

def refine_mask(mask, kernel_size=5, blur_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(opened, (blur_size, blur_size), sigmaX=1)
    return blurred
