import cv2
import numpy as np
from ultralytics import YOLO

# 載入模型與影像
model = YOLO("yolov11-seg.pt")
results = model.predict("your_image.jpg")

result = results[0]
img = result.orig_img.copy()
H, W = img.shape[:2]

# 取得第一個 instance mask（你可迴圈跑多個）
if result.masks is not None:
    mask_tensor = result.masks.data[0]  # shape: [640, 640]
    mask = mask_tensor.cpu().numpy()

    # resize mask 回原圖大小
    mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized > 0.5).astype(np.uint8)

    # 建立上下邊界陣列
    upper = np.full(W, -1)
    lower = np.full(W, -1)

    for x in range(W):
        col = mask_bin[:, x]
        y_indices = np.where(col == 1)[0]
        if len(y_indices) > 0:
            upper[x] = y_indices[0]
            lower[x] = y_indices[-1]

    # 畫線：上邊界為紅色，下邊界為藍色
    for x in range(1, W):
        if upper[x] > 0 and upper[x-1] > 0:
            cv2.line(img, (x-1, upper[x-1]), (x, upper[x]), (0, 0, 255), 2)
        if lower[x] > 0 and lower[x-1] > 0:
            cv2.line(img, (x-1, lower[x-1]), (x, lower[x]), (255, 0, 0), 2)

    # 顯示結果
    cv2.imshow("Top & Bottom Edge", cv2.resize(img, (640, 360)))
    cv2.waitKey(0)
