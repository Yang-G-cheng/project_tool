# yolo_sam_pipeline.py
# Requires: pip install ultralytics segment-anything opencv-python

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2
import torch
import numpy as np

# 1. Load image
img_path = 'input.jpg'  # Replace with your image
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Run YOLOv11 detection
model = YOLO('yolov8n.pt')  # or your custom YOLOv11 model
results = model.predict(image_rgb)
bboxes = results[0].boxes.xyxy.cpu().numpy()

# 3. Load SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")  # download from Meta AI
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# 4. Use YOLO bbox as SAM input
for i, box in enumerate(bboxes):
    masks, _, _ = predictor.predict(box=box.astype(int), multimask_output=True)
    mask = masks[0]

    # Overlay mask on image
    vis = image.copy()
    vis[mask] = (0, 255, 0)
    cv2.imshow(f"YOLO+SAM Mask {i}", cv2.resize(vis, (640, 360)))
    cv2.waitKey(0)
