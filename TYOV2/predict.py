from ultralytics import YOLO
from pathlib import Path
import cv2
use_webcam = False
model = YOLO("yolo11n-seg.pt");use_source = 'C:\\Users\\YAN\\Pictures\\yolo\\inference'
out_dir = 'C:\\Users\\YAN\Pictures\\yolo\\test_auto_label'
name='exp';scale = 0.5
# source = 
# model.predict(source, save = True)
# 開啟 webcam
if use_webcam:
    cap = cv2.VideoCapture(0)  # 0 表示預設 webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # 即時處理迴圈
    while True:
        ret, frame = cap.read()  # 讀取一幀
        if not ret:
            print("Error: Failed to capture image.")
            break

        # 使用 YOLOv11 進行推論
        results = model.predict(source=frame, conf=0.25, imgsz=640)

        # 將結果繪製到影像上
        annotated_frame = results[0].plot()  # 繪製邊界框、類別等

        # 顯示結果
        cv2.imshow("YOLOv11 Webcam", annotated_frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
else:
    def save_labelme_results(image_path, det, names, save_dir):
        import base64
        import io
        import json

        from PIL import Image

        def img_to_b64(image):
            f = io.BytesIO()
            image.save(f, format="PNG")
            img_bin = f.getvalue()
            return base64.encodebytes(img_bin) if hasattr(base64, "encodebytes") else base64.encodestring(img_bin)

        image = Image.open(image_path)
        imageWidth, imageHeight = image.size
        imagePath, fileName = str(image_path), image_path.stem

        annotion = {
            "version": "5.0.5", 
            "flags": {}, 
            "shapes": [], 
            "imagePath": imagePath, 
            "imageData": img_to_b64(image).decode('ascii'), 
            "imageHeight": imageHeight,
            "imageWidth": imageWidth,
        }

        annotion["shapes"].extend([{
            "label": names[int(cls_id)], # clss label
            "points": [[int(xyxy[0]), int(xyxy[1])], [int(xyxy[2]), int(xyxy[3])]], # bounding box
            "group_id": None, 
            "shape_type": "rectangle", 
            "flags": {}
        } for *xyxy, _, cls_id in reversed(det)])

        json.dump(
            annotion, 
            open(Path(save_dir, f'{fileName}.json'), "w+", encoding='utf-8'), 
            indent=4, sort_keys=False, ensure_ascii=False
        )  # 保存json
    source = use_source
    results = model.predict(source, save = True, project='custom_output' ,name='webcam_results')
    # print(results.masks.data.shape)
    # for r in results:
        # print(r.masks.data.shape)  # (N, H, W) segmentation masks torch.Size([1, 640, 320])
        # print(r.masks.xyn)      #list of polygons (normalized)
        # print(r.boxes.xyxy)    # (N, 4) tensor: x1, y1, x2, y2
        # print(r.boxes.conf)    # (N,) confidence
        # print(r.boxes.cls)     # (N,) class indices
    for result in results:
        print(result)
    #     img = result.orig_img
    #     # img = cv2.resize(img, (640, 320))
    #     h, w = img.shape[:2]
    

    #     for i, mask_tensor in enumerate(result.masks.data):
    #         mask = mask_tensor.cpu().numpy()
    #         mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    #         # 將 mask 疊加到原圖上（例如塗綠）
    #         img_masked = img.copy()
    #         img_masked[mask_resized > 0.5] = [0, 255, 0]
    #         img_small = cv2.resize(img_masked, (int(w * scale), int(h * scale)))
    #         cv2.imshow(f"Seg {i}", img_masked)
    #         cv2.waitKey(0)
    # # from pathlib import Path
    #     save_labelme_results(source, result, name, out_dir)

    