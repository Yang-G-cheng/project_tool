import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

def ransac_smooth_contour(contour, max_trials=100, min_samples=10):
    """
    使用 RANSAC 擬合輪廓點為平滑的直線段
    Args:
        contour: OpenCV 輪廓點 (numpy array of shape [n, 1, 2])
        max_trials: RANSAC 最大迭代次數
        min_samples: RANSAC 最小樣本數
    Returns:
        smoothed_contour: 平滑後的輪廓點
    """
    # 將輪廓點轉為 (x, y) 坐標
    points = contour.reshape(-1, 2)
    x = points[:, 0].reshape(-1, 1)  # x 坐標
    y = points[:, 1]  # y 坐標

    # 使用 RANSAC 擬合直線
    ransac = RANSACRegressor(max_trials=max_trials, min_samples=min_samples, residual_threshold=5.0)
    ransac.fit(x, y)

    # 獲取內點
    inlier_mask = ransac.inlier_mask_
    inlier_points = points[inlier_mask]

    # 用擬合的直線生成平滑點
    x_range = np.linspace(x.min(), x.max(), len(points)).reshape(-1, 1)
    y_pred = ransac.predict(x_range)
    smoothed_contour = np.column_stack((x_range, y_pred)).astype(np.int32)

    return smoothed_contour.reshape(-1, 1, 2)

def post_process_mask_with_ransac(mask, min_contour_area=100):
    """
    對 YOLOv7-seg 的分割遮罩進行後處理，使用 RANSAC 平滑輪廓
    Args:
        mask: 二值化遮罩 (numpy array, 0 或 255)
        min_contour_area: 最小輪廓面積閾值
    Returns:
        smoothed_mask: 平滑後的遮罩
    """
    # 確保遮罩是二值化的
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 提取輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed_mask = np.zeros_like(mask)

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # 使用 RANSAC 平滑輪廓
            smoothed_contour = ransac_smooth_contour(contour)
            cv2.drawContours(smoothed_mask, [smoothed_contour], -1, (255), -1)

    return smoothed_mask

# 示例：假設有一個 YOLOv7-seg 的遮罩輸出
def main():
    # 模擬一個 YOLOv7-seg 的分割遮罩
    # mask = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    mask = cv2.imread("test_edge.png", 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 進行 RANSAC 後處理
    smoothed_mask = post_process_mask_with_ransac(
        mask,
        min_contour_area=100
    )

    # 保存或顯示結果
    # cv2.imwrite('original_mask.png', mask)
    # cv2.imwrite('ransac_smoothed_mask.png', smoothed_mask)

    # 可選：顯示遮罩
    cv2.imshow('Original Mask', mask)
    cv2.imshow('RANSAC Smoothed Mask', smoothed_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()