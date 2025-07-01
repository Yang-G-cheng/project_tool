import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN

def analyze_wafer_mask(mask_data, image_path=None):
    """
    分析YOLOv11輸出的wafer mask資料，找出邊緣上下邊界並計算距離
    
    Args:
        mask_data: YOLOv11輸出的mask資料 (numpy array或二進位mask)
        image_path: 原始圖片路徑 (可選，用於視覺化)
    
    Returns:
        dict: 包含上下邊界座標和距離資訊
    """
    
    # 確保mask是二進位格式
    if mask_data.dtype != np.uint8:
        mask_data = (mask_data * 255).astype(np.uint8)
    
    # 如果mask不是二進位，進行閾值處理
    if len(np.unique(mask_data)) > 2:
        _, mask_data = cv2.threshold(mask_data, 127, 255, cv2.THRESH_BINARY)
    
    return analyze_mask_boundaries(mask_data)

def analyze_mask_boundaries(binary_mask):
    """
    分析二進位mask的邊界
    """
    # 找到mask的輪廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"error": "未找到有效輪廓"}
    
    # 選擇最大的輪廓（假設是wafer的主要輪廓）
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 獲取輪廓的所有點
    contour_points = largest_contour.reshape(-1, 2)
    
    # 分析上下邊界
    results = analyze_vertical_boundaries(contour_points, binary_mask)
    
    return results

def analyze_vertical_boundaries(contour_points, mask):
    """
    分析垂直邊界（上邊界和下邊界）
    """
    # 找到最高點和最低點
    top_y = np.min(contour_points[:, 1])
    bottom_y = np.max(contour_points[:, 1])
    
    # 獲取圖像的寬度
    height, width = mask.shape
    
    # 逐列掃描找到上下邊界
    upper_boundary = []
    lower_boundary = []
    
    for x in range(width):
        # 獲取當前列的所有像素
        column = mask[:, x]
        
        # 找到該列中mask為255的像素位置
        white_pixels = np.where(column == 255)[0]
        
        if len(white_pixels) > 0:
            # 上邊界是該列最小的y座標
            upper_y = np.min(white_pixels)
            # 下邊界是該列最大的y座標
            lower_y = np.max(white_pixels)
            
            upper_boundary.append((x, upper_y))
            lower_boundary.append((x, lower_y))
    
    if not upper_boundary or not lower_boundary:
        return {"error": "未找到有效邊界"}
    
    # 計算邊界品質和距離資訊
    results = calculate_boundary_metrics(upper_boundary, lower_boundary, mask.shape)
    
    return results

def calculate_boundary_metrics(upper_boundary, lower_boundary, image_shape):
    """
    計算邊界的品質指標和距離資訊
    """
    upper_boundary = np.array(upper_boundary)
    lower_boundary = np.array(lower_boundary)
    
    # 計算每個x位置的上下邊界距離
    distances = []
    for i in range(len(upper_boundary)):
        if i < len(lower_boundary):
            dist = lower_boundary[i][1] - upper_boundary[i][1]
            distances.append(dist)
    
    distances = np.array(distances)
    
    # 計算統計資訊
    stats = {
        "mean_distance": np.mean(distances),
        "std_distance": np.std(distances),
        "min_distance": np.min(distances),
        "max_distance": np.max(distances),
        "median_distance": np.median(distances)
    }
    
    # 分析邊界的平滑度
    upper_smoothness = calculate_smoothness(upper_boundary[:, 1])
    lower_smoothness = calculate_smoothness(lower_boundary[:, 1])
    
    # 檢測異常點
    upper_outliers = detect_outliers(upper_boundary[:, 1])
    lower_outliers = detect_outliers(lower_boundary[:, 1])
    
    # 評估圓形度（對於wafer應該是圓形）
    circularity = evaluate_circularity(upper_boundary, lower_boundary)
    
    results = {
        "upper_boundary": upper_boundary.tolist(),
        "lower_boundary": lower_boundary.tolist(),
        "distance_stats": stats,
        "upper_smoothness": upper_smoothness,
        "lower_smoothness": lower_smoothness,
        "upper_outliers": upper_outliers,
        "lower_outliers": lower_outliers,
        "circularity_score": circularity,
        "boundary_quality": assess_boundary_quality(stats, upper_smoothness, lower_smoothness)
    }
    
    return results

def calculate_smoothness(boundary_points):
    """
    計算邊界的平滑度（使用梯度變化）
    """
    if len(boundary_points) < 3:
        return 0
    
    # 計算一階導數（梯度）
    gradients = np.gradient(boundary_points)
    
    # 計算二階導數（曲率）
    curvatures = np.gradient(gradients)
    
    # 平滑度 = 1 / (1 + 曲率方差)
    smoothness = 1 / (1 + np.var(curvatures))
    
    return float(smoothness)

def detect_outliers(points, threshold=2.0):
    """
    使用Z-score檢測異常點
    """
    if len(points) < 3:
        return []
    
    z_scores = np.abs((points - np.mean(points)) / np.std(points))
    outlier_indices = np.where(z_scores > threshold)[0]
    
    return outlier_indices.tolist()

def evaluate_circularity(upper_boundary, lower_boundary):
    """
    評估邊界的圓形度
    """
    # 合併上下邊界點
    all_points = np.vstack([upper_boundary, lower_boundary])
    
    # 計算中心點
    center_x = np.mean(all_points[:, 0])
    center_y = np.mean(all_points[:, 1])
    
    # 計算到中心的距離
    distances = np.sqrt((all_points[:, 0] - center_x)**2 + (all_points[:, 1] - center_y)**2)
    
    # 圓形度 = 1 - (距離標準差 / 平均距離)
    circularity = 1 - (np.std(distances) / np.mean(distances))
    
    return float(max(0, circularity))  # 確保非負

def assess_boundary_quality(distance_stats, upper_smoothness, lower_smoothness):
    """
    綜合評估邊界品質
    """
    # 距離一致性 (標準差越小越好)
    distance_consistency = 1 / (1 + distance_stats["std_distance"] / distance_stats["mean_distance"])
    
    # 邊界平滑度
    avg_smoothness = (upper_smoothness + lower_smoothness) / 2
    
    # 綜合品質分數
    quality_score = (distance_consistency * 0.4 + avg_smoothness * 0.6)
    
    quality_level = "優秀" if quality_score > 0.8 else "良好" if quality_score > 0.6 else "一般" if quality_score > 0.4 else "較差"
    
    return {
        "score": float(quality_score),
        "level": quality_level,
        "distance_consistency": float(distance_consistency),
        "smoothness": float(avg_smoothness)
    }

def visualize_results(original_image, mask, results):
    """
    視覺化分析結果
    """
    plt.figure(figsize=(15, 10))
    
    # 原圖
    plt.subplot(2, 3, 1)
    if original_image is not None:
        plt.imshow(original_image, cmap='gray')
    plt.title('原始圖像')
    
    # Mask
    plt.subplot(2, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('YOLOv11 Mask')
    
    # 邊界視覺化
    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap='gray', alpha=0.5)
    
    if 'upper_boundary' in results and 'lower_boundary' in results:
        upper_boundary = np.array(results['upper_boundary'])
        lower_boundary = np.array(results['lower_boundary'])
        
        plt.plot(upper_boundary[:, 0], upper_boundary[:, 1], 'r-', linewidth=2, label='上邊界')
        plt.plot(lower_boundary[:, 0], lower_boundary[:, 1], 'b-', linewidth=2, label='下邊界')
        
        # 標記異常點
        if results['upper_outliers']:
            outlier_points = upper_boundary[results['upper_outliers']]
            plt.scatter(outlier_points[:, 0], outlier_points[:, 1], c='red', s=50, marker='x')
        
        if results['lower_outliers']:
            outlier_points = lower_boundary[results['lower_outliers']]
            plt.scatter(outlier_points[:, 0], outlier_points[:, 1], c='blue', s=50, marker='x')
    
    plt.legend()
    plt.title('邊界檢測結果')
    
    # 距離分佈
    plt.subplot(2, 3, 4)
    if 'upper_boundary' in results and 'lower_boundary' in results:
        upper_boundary = np.array(results['upper_boundary'])
        lower_boundary = np.array(results['lower_boundary'])
        
        distances = []
        for i in range(min(len(upper_boundary), len(lower_boundary))):
            dist = lower_boundary[i][1] - upper_boundary[i][1]
            distances.append(dist)
        
        plt.plot(distances)
        plt.axhline(y=np.mean(distances), color='r', linestyle='--', label=f'平均距離: {np.mean(distances):.2f}')
        plt.xlabel('X座標')
        plt.ylabel('上下邊界距離(像素)')
        plt.title('上下邊界距離分佈')
        plt.legend()
    
    # 品質指標
    plt.subplot(2, 3, 5)
    if 'boundary_quality' in results:
        quality = results['boundary_quality']
        metrics = ['總體品質', '距離一致性', '邊界平滑度']
        values = [quality['score'], quality['distance_consistency'], quality['smoothness']]
        
        bars = plt.bar(metrics, values, color=['green', 'blue', 'orange'])
        plt.ylim(0, 1)
        plt.title('邊界品質評估')
        plt.xticks(rotation=45)
        
        # 添加數值標籤
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 統計資訊
    plt.subplot(2, 3, 6)
    plt.axis('off')
    if 'distance_stats' in results:
        stats = results['distance_stats']
        info_text = f"""
距離統計資訊:
平均距離: {stats['mean_distance']:.2f} 像素
標準差: {stats['std_distance']:.2f} 像素
最小距離: {stats['min_distance']:.2f} 像素
最大距離: {stats['max_distance']:.2f} 像素
中位數: {stats['median_distance']:.2f} 像素

圓形度評分: {results.get('circularity_score', 0):.3f}
邊界品質: {results.get('boundary_quality', {}).get('level', 'N/A')}
        """
        plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

# 使用範例
def example_usage():
    """
    使用範例
    """
    # 範例1: 從YOLOv11結果檔案載入
    # results = model.predict('wafer_image.jpg')  # YOLOv11預測
    # mask = results[0].masks.data[0].cpu().numpy()  # 獲取mask資料
    
    # 範例2: 直接分析mask陣列
    # 假設你已經有了mask資料
    # analysis_results = analyze_wafer_mask(mask_data)
    # print("分析結果:", analysis_results)
    
    # 範例3: 視覺化結果
    # visualize_results(original_image, mask_data, analysis_results)
    
    print("請使用以下函數來分析您的mask資料:")
    print("1. analyze_wafer_mask(mask_data) - 主要分析函數")
    print("2. visualize_results(original_image, mask, results) - 視覺化結果")

if __name__ == "__main__":
    example_usage()