import os
import json
import base64
import io
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

def img_to_b64(image):
    f = io.BytesIO()
    image.save(f, format="PNG")
    return base64.encodebytes(f.getvalue()).decode('ascii')

def save_labelme_segmentation(image_path, result, names, save_dir):
    image = Image.open(image_path)
    imageWidth, imageHeight = image.size
    fileName = Path(image_path).stem

    annotation = {
        "version": "5.0.5",
        "flags": {},
        "shapes": [],
        "imagePath": str(Path(image_path).name),
        "imageData": img_to_b64(image),
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
    }

    masks = result.masks.xyn if result.masks else []
    classes = result.boxes.cls.tolist() if result.boxes else []

    for poly, cls_id in zip(masks, classes):
        points = [[int(x * imageWidth), int(y * imageHeight)] for x, y in poly]
        annotation["shapes"].append({
            "label": names[int(cls_id)],
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

    out_path = Path(save_dir) / f"{fileName}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=4, ensure_ascii=False)

def batch_convert_yolo_to_labelme(model_path, image_dir, output_dir):
    model = YOLO(model_path)
    names = model.names
    image_paths = list(Path(image_dir).glob("*.[jp][pn]g"))  # jpg or png

    os.makedirs(output_dir, exist_ok=True)
    results = model.predict(image_paths, save=False)

    for img_path, result in zip(image_paths, results):
        save_labelme_segmentation(img_path, result, names, output_dir)

if __name__ == "__main__":
    model_path = "yolo11n-seg.pt"
    image_dir = "images"
    output_dir = "labelme_json"

    batch_convert_yolo_to_labelme(model_path, image_dir, output_dir)
