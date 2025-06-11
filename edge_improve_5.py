import cv2
import numpy as np
import torch
from sklearn.linear_model import RANSACRegressor
from models.experimental import attempt_load  # YOLOv7 的模型加載函數
from utils.general import non_max_suppression, scale_coords
from utils.segment.general import process_mask, scale_masks

def load_yolov7_model(weights_path, device='cuda'):
    """加載 YOLOv7-seg 模型"""
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    return model

def inference_yolov7_seg(model, img, conf_thres=0.25, iou_thres=0.45, device='cuda'):
    """
    對完整圖像進行 YOLOv7-seg 推理
    Args:
        model: YOLOv7-seg 模型
        img: 輸入圖像 (numpy array, BGR)
        conf_thres: 置信度閾值
        iou_thres: NMS IoU 閾值
    Returns:
        masks: 分割遮罩 (numpy array, shape: [H, W])
    """
    # 圖像預處理
    img_orig = img.copy()
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, (640, 640))  # YOLOv7 標準輸入大小
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)  # Add batch dimension

    # 推理
    with torch.no_grad():
        pred, proto = model(img)[1:]  # YOLOv7-seg 輸出

    # 非最大抑制
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

    # 處理分割遮罩
    masks = None
    for det in pred:
        if len(det):
            # 縮放檢測框到原始圖像尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_orig.shape).round()
            # 生成遮罩
            masks = process_mask(proto[0], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)
            masks = scale_masks(img.shape[2:], masks, img_orig.shape)  # 縮放到原始尺寸
            masks = masks.cpu().numpy()  # 轉為 numpy
            break  # 假設只處理第一個批次

    # 如果沒有檢測到對象，返回全零遮罩
    if masks is None:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # 合併所有遮罩
    final_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    return final_mask

def ransac_smooth_contour(contour, max_trials=100, min_samples=10):
    """使用 RANSAC 擬合輪廓點"""
    points = contour.reshape(-1, 2)
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    ransac = RANSACRegressor(max_trials=max_trials, min_samples=min_samples, residual_threshold=5.0)
    ransac.fit(x, y)
    x_range = np.linspace(x.min(), x.max(), len(points)).reshape(-1, 1)
    y_pred = ransac.predict(x_range)
    smoothed_contour = np.column_stack((x_range, y_pred)).astype(np.int32)
    return smoothed_contour.reshape(-1, 1, 2)

def post_process_mask_with_ransac(mask, morph_kernel_size=3, gaussian_sigma=1.0, min_contour_area=50):
    """
    後處理遮罩，確保完整輸出
    """
    # 形態學處理
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 高斯模糊
    mask = cv2.GaussianBlur(mask, (5, 5), gaussian_sigma)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # RANSAC 平滑輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed_mask = np.zeros_like(mask)

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            smoothed_contour = ransac_smooth_contour(contour)
            cv2.drawContours(smoothed_mask, [smoothed_contour], -1, (255), -1)

    return smoothed_mask

def main():
    # # 參數設置
    # weights_path = 'path/to/yolov7-seg.pt'  # 替換為你的模型權重路徑
    # img_path = 'path/to/your/image.jpg'  # 替換為你的圖像路徑
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # 加載模型
    # model = load_yolov7_model(weights_path, device)

    # # 讀取圖像
    # img = cv2.imread(img_path)
    # if img is None:
    #     print("Error: Cannot load image.")
    #     return

    # # 推理
    # mask = inference_yolov7_seg(model, img, conf_thres=0.2, iou_thres=0.4, device=device)
    mask = cv2.imread("test_edge.png", 0)
    # 後處理
    smoothed_mask = post_process_mask_with_ransac(
        mask,
        morph_kernel_size=3,
        gaussian_sigma=1.0,
        min_contour_area=50
    )

    # 保存結果
    cv2.imwrite('original_mask.png', mask)
    cv2.imwrite('smoothed_mask.png', smoothed_mask)

    # 顯示結果
    cv2.imshow('Original Mask', mask)
    cv2.imshow('Smoothed Mask', smoothed_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()