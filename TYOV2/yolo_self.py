from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
# results = model.train(data="coco8.yaml", epochs=5)
model.train(data='my_seg.yaml', epochs=10, imgsz=640, pretrained=False)
# model.train(
#     data='my_seg.yaml',        # dataset 描述檔
#     epochs=3,
#     imgsz=640,
#     batch=16,
#     device=0,                  # 0 表 GPU，'cpu' 表 CPU
#     workers=8,
#     pretrained=False,          # 不使用預訓練權重
#     optimizer='SGD',           # 可改 AdamW 或其他
#     lr0=0.01,                  # 初始學習率
#     project='runs/train',
#     name='yolov11_seg_from_scratch',
#     verbose=True
# )