# utils.py
import cv2
import os

def save_masks(masks, base_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, m in enumerate(masks):
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_{i}.png"), m)
