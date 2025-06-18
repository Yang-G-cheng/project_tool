# main.py
import argparse
import os
from glob import glob
import cv2

from edge_segmentation import HybridEdgeDetector, refine_mask, save_masks, generate_yolo_labels

def process_images(yolo_model_path, img_dir, mask_dir, label_dir):
    print("🚀 初始化 YOLOv11 + Mask R-CNN 模型...")
    detector = HybridEdgeDetector(yolo_model_path)

    image_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    os.makedirs(mask_dir, exist_ok=True)

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"🔍 處理中: {img_name}")

        # Step 1: YOLOv11 + Mask R-CNN 雙階段偵測
        masks = detector.detect(img_path)

        # Step 2: 邊緣後處理
        refined_masks = [refine_mask(m) for m in masks]

        # Step 3: 儲存 masks
        save_masks(refined_masks, img_name, mask_dir)

    # Step 4: 自動產生 YOLO segment 標籤
    print("🏷️ 產生 YOLO segmentation label...")
    generate_yolo_labels(img_dir, mask_dir, label_dir)

    print("✅ 完成所有處理！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 + Mask R-CNN 邊緣精準 segmentation pipeline")
    parser.add_argument("--yolo", type=str, required=True, help="YOLOv11 權重路徑")
    parser.add_argument("--img_dir", type=str, default="data/images", help="輸入圖片資料夾")
    parser.add_argument("--mask_dir", type=str, default="data/masks", help="輸出 mask 資料夾")
    parser.add_argument("--label_dir", type=str, default="data/labels", help="輸出 YOLO 格式 label 資料夾")

    args = parser.parse_args()
    process_images(args.yolo, args.img_dir, args.mask_dir, args.label_dir)
