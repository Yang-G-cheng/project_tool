# detector.py
import cv2
import numpy as np
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

class HybridEdgeDetector:
    def __init__(self, yolo_model_path: str, maskrcnn_weights_url: str = None):
        self.yolo = YOLO(yolo_model_path)

        # 初始化 Mask R-CNN
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        if maskrcnn_weights_url:
            cfg.MODEL.WEIGHTS = maskrcnn_weights_url
        else:
            cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.mask_predictor = DefaultPredictor(cfg)

    def detect(self, image_path: str):
        img = cv2.imread(image_path)
        results = self.yolo(img, task="detect")[0]
        boxes = results.boxes.xyxy.cpu().numpy()

        masks = []
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            roi = img[y1:y2, x1:x2]
            output = self.mask_predictor(roi)
            if len(output["instances"]) == 0:
                continue
            mask = output["instances"].pred_masks[0].cpu().numpy()
            full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2][mask] = 255
            masks.append(full_mask)
        return masks
