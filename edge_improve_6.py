import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.segment.general import process_mask, scale_masks

def load_yolov7_model(weights_path, device='cuda'):
    """加載 YOLOv7-seg 模型"""
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    return model

def inference_yolov7_seg(model, img, conf_thres=0.2, iou_thres=0.4, device='cuda'):
    """
    對完整圖像進行 YOLOv7-seg 推理，確保遮罩與原始圖像尺寸匹配
    Args:
        model: YOLOv7-seg 模型
        img: 輸入圖像 (numpy array, BGR)
        conf_thres: 置信度閾值
        iou_thres: NMS IoU 閾值
    Returns:
        masks: 分割遮罩 (numpy array, shape: [H, W])
    """
    # 保存原始圖像尺寸
    img_orig = img.copy()
    img_h, img_w = img.shape[:2]
    print(f"Original image size: {img_w}x{img_h}")

    # 圖像預處理（縮放到 YOLOv7 輸入尺寸）
    img_input = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img_input = img_input.transpose(2, 0, 1)  # HWC to CHW
    img_input = torch.from_numpy(img_input).to(device).float() / 255.0
    img_input = img_input.unsqueeze(0)  # Add batch dimension

    # 推理
    with torch.no_grad():
        pred, proto = model(img_input)[1:]  # YOLOv7-seg 輸出

    # 非最大抑制
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    
    # 處理分割遮罩
    masks = None
    for det in pred:
        if len(det):
            # 縮放檢測框到原始圖像尺寸
            det[:, :4] = scale_coords(img_input.shape[2:], det[:, :4], img_orig.shape).round()
            # 生成遮罩並縮放到原始尺寸
            masks = process_mask(proto[0], det[:, 6:], det[:, :4], img_input.shape[2:], upsample=True)
            masks = scale_masks(img_input.shape[2:], masks, img_orig.shape)  # 縮放到原始尺寸
            masks = masks.cpu().numpy()
            print(f"Generated masks shape: {masks.shape}")
            break

    # 如果沒有檢測到對象，返回全零遮罩
    if masks is None:
        print("No objects detected, returning empty mask.")
        return np.zeros((img_h, img_w), dtype=np.uint8)

    # 合併所有遮罩
    final_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    print(f"Final mask shape: {final_mask.shape}")
    return final_mask

def post_process_mask(mask, morph_kernel_size=3, gaussian_sigma=1.0, min_contour_area=50):
    """
    後處理遮罩，禁用 RANSAC，僅使用形態學和高斯模糊
    Args:
        mask: 二值化遮罩 (numpy array, 0 或 255)
        morph_kernel_size: 形態學核大小
        gaussian_sigma: 高斯模糊標準差
        min_contour_area: 最小輪廓面積
    Returns:
        smoothed_mask: 平滑後的遮罩
    """
    print(f"Input mask shape for post-processing: {mask.shape}")

    # 確保遮罩是二值化的
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 形態學處理
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 去除噪點
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填補小孔

    # 高斯模糊平滑邊緣
    mask = cv2.GaussianBlur(mask, (5, 5), gaussian_sigma)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 輪廓平滑化（使用多邊形近似代替 RANSAC）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed_mask = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # 多邊形近似
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(smoothed_mask, [approx], -1, (255), -1)

    print(f"Smoothed mask shape: {smoothed_mask.shape}")
    return smoothed_mask

def main():
    # 參數設置
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
    smoothed_mask = post_process_mask(
        mask,
        morph_kernel_size=3,
        gaussian_sigma=1.0,
        min_contour_area=50
    )

    # 保存結果
    # cv2.imwrite('original_mask.png', mask)
    # cv2.imwrite('smoothed_mask.png', smoothed_mask)

    # 顯示結果
    cv2.imshow('Original Mask', mask)
    cv2.imshow('Smoothed Mask', smoothed_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()