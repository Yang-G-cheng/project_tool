import numpy as np
import cv2

# 假設你已經取得了 mask，例如：
# mask = np.zeros((H, W), dtype=np.uint8)
# mask[y, x] = 1 對於前景點
# 或 mask 是從模型輸出來的 segmentation binary mask

# 找出所有前景點的 (x, y)
ys, xs = np.where(mask > 0)
points = list(zip(xs, ys))  # (x, y)

# 把這些點按 x 排序
points.sort(key=lambda p: p[0])  # 對 x 排序

# 建立 dict，每個 x 對應 min_y（上邊界）、max_y（下邊界）
x_dict = {}
for x, y in points:
    if x not in x_dict:
        x_dict[x] = {'min_y': y, 'max_y': y}
    else:
        x_dict[x]['min_y'] = min(x_dict[x]['min_y'], y)
        x_dict[x]['max_y'] = max(x_dict[x]['max_y'], y)

# 取得上邊界和下邊界點列表
upper_boundary = [(x, val['min_y']) for x, val in x_dict.items()]
lower_boundary = [(x, val['max_y']) for x, val in x_dict.items()]

# 載入原始圖像（要畫線）
img = cv2.imread('your_image.png')

# 畫上邊界：綠色
for pt1, pt2 in zip(upper_boundary[:-1], upper_boundary[1:]):
    cv2.line(img, pt1, pt2, (0, 255, 0), 2)

# 畫下邊界：紅色
for pt1, pt2 in zip(lower_boundary[:-1], lower_boundary[1:]):
    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

# 顯示
cv2.imshow("Mask Boundaries", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
