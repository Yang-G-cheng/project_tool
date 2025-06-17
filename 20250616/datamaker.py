import os
import random
from PIL import Image, ImageDraw
from pathlib import Path

# Define dataset paths
base_path = Path("D:\\程式碼\\回家後\\yolov7\\yolo_self\\my_seg_dataset")
image_train_path = base_path / "images" / "train"
image_val_path = base_path / "images" / "val"
label_train_path = base_path / "labels" / "train"
label_val_path = base_path / "labels" / "val"

# Create directories
for path in [image_train_path, image_val_path, label_train_path, label_val_path]:
    path.mkdir(parents=True, exist_ok=True)

# Define a mock function to create polygon mask label and image
def create_mock_sample(img_path, label_path, num_objects=2):
    img = Image.new("RGB", (640, 640), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    with open(label_path, "w") as f:
        for _ in range(num_objects):
            cls_id = random.randint(0, 2)
            polygon = []
            for _ in range(6):  # 6 points polygon
                x = random.randint(50, 590)
                y = random.randint(50, 590)
                polygon.extend([x, y])
            draw.polygon(polygon, outline=(255, 0, 0))
            norm_polygon = [str(p / 640) for p in polygon]
            f.write(f"{cls_id} " + " ".join(norm_polygon) + "\n")
    img.save(img_path)

# Generate mock data
for i in range(10):
    create_mock_sample(image_train_path / f"img_{i}.jpg", label_train_path / f"img_{i}.txt")
for i in range(5):
    create_mock_sample(image_val_path / f"img_{i}.jpg", label_val_path / f"img_{i}.txt")

# # Return the base path to inform user
print(base_path)
