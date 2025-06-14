from pathlib import Path

label_root = Path("D:\\程式碼\回家後\\yolov7\\yolo_self\\my_seg_dataset\\labels\\train")
errors = []

for label_file in label_root.glob("*.txt"):
    with open(label_file, "r") as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                errors.append((label_file.name, i, "Empty line"))
                continue

            cls, *coords = parts
            try:
                cls_id = int(cls)
                coords = list(map(float, coords))
            except ValueError:
                errors.append((label_file.name, i, "Non-numeric class or coordinates"))
                continue

            if len(coords) % 2 != 0:
                errors.append((label_file.name, i, "Odd number of polygon coordinates"))

            if not all(0 <= x <= 1 for x in coords):
                errors.append((label_file.name, i, "Coordinate out of [0, 1] range"))

for err in errors:
    print("❌", err)
