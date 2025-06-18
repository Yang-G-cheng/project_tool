# label_generator.py
import os
import cv2
import numpy as np

def generate_yolo_labels(image_dir, mask_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    for fname in os.listdir(image_dir):
        name = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, name + ".png")
        if not os.path.exists(mask_path): continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        mask = cv2.imread(mask_path, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_lines = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            norm_bw = bw / w
            norm_bh = bh / h
            label_lines.append(f"0 {xc:.6f} {yc:.6f} {norm_bw:.6f} {norm_bh:.6f}")

        with open(os.path.join(label_dir, name + ".txt"), "w") as f:
            f.write("\n".join(label_lines))
