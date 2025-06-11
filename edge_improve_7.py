import cv2
import numpy as np

def post_process_mask(mask, morph_kernel_size=5, gaussian_sigma=1.0, epsilon_factor=0.01):
    """
    對 YOLOv7-seg 的分割遮罩進行後處理以平滑邊緣
    Args:
        mask: 二值化遮罩 (numpy array, 0 或 255)
        morph_kernel_size: 形態學核大小
        gaussian_sigma: 高斯模糊標準差
        epsilon_factor: 多邊形近似的 epsilon 因子 (相對於弧長的比例)
    Returns:
        smoothed_mask: 平滑後的遮罩
    """
    # 確保遮罩是二值化的
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 1. 形態學處理：去除噪點並填補小孔
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2. 高斯模糊平滑邊緣
    mask = cv2.GaussianBlur(mask, (5, 5), gaussian_sigma)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. 輪廓平滑化
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed_mask = np.zeros_like(mask)

    for contour in contours:
        # 計算弧長並應用多邊形近似
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(smoothed_mask, [approx], -1, (255), -1)

    return smoothed_mask

# 示例：處理你的預測結果
def main():
    # 假設這是你從 YOLOv7-seg 得到的遮罩（二值化後的 numpy 陣列）
    # 這裡使用圖片模擬你的預測結果
    img = cv2.imread("test_edge.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Cannot load image.")
        return

    # 進行後處理
    smoothed_mask = post_process_mask(
        img,
        morph_kernel_size=5,
        gaussian_sigma=1.0,
        epsilon_factor=0.01
    )

    # 保存結果
    # cv2.imwrite('smoothed_mask.png', smoothed_mask)

    # 顯示結果
    cv2.imshow('Original Mask', img)
    cv2.imshow('Smoothed Mask', smoothed_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()