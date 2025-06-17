import cv2
import numpy as np
from ultralytics import YOLO
from scipy.interpolate import interp1d
import math

class ArcBoundaryAnalyzer:
    def __init__(self):
        self.upper_boundary = None
        self.lower_boundary = None
        self.upper_func = None
        self.lower_func = None
        self.x_range = None
        
    def extract_arc_boundaries(self, contour, img_shape):
        """
        從輪廓中提取上下邊界
        """
        if len(contour) < 5:
            return None, None
            
        # 將輪廓轉換為點陣列
        points = contour.reshape(-1, 2)
        
        # 按x座標排序
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        # 找到每個x座標對應的最高點和最低點
        x_coords = sorted_points[:, 0]
        y_coords = sorted_points[:, 1]
        
        # 創建x座標的唯一值範圍
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        x_range = np.arange(x_min, x_max + 1)
        
        upper_points = []
        lower_points = []
        
        # 對每個x座標找到對應的上下邊界點
        for x in x_range:
            # 找到該x座標附近的所有點 (容許±2像素的誤差)
            mask = np.abs(points[:, 0] - x) <= 2
            if np.any(mask):
                y_candidates = points[mask][:, 1]
                upper_y = np.min(y_candidates)  # 最小y值(上邊界)
                lower_y = np.max(y_candidates)  # 最大y值(下邊界)
                
                upper_points.append([x, upper_y])
                lower_points.append([x, lower_y])
        
        if len(upper_points) < 5 or len(lower_points) < 5:
            return None, None
            
        upper_boundary = np.array(upper_points)
        lower_boundary = np.array(lower_points)
        
        # 平滑處理邊界點
        upper_boundary = self._smooth_boundary(upper_boundary)
        lower_boundary = self._smooth_boundary(lower_boundary)
        
        # 創建插值函數
        self.upper_func = interp1d(upper_boundary[:, 0], upper_boundary[:, 1], 
                                  kind='cubic', bounds_error=False, fill_value='extrapolate')
        self.lower_func = interp1d(lower_boundary[:, 0], lower_boundary[:, 1], 
                                  kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        self.x_range = (x_min, x_max)
        self.upper_boundary = upper_boundary
        self.lower_boundary = lower_boundary
        
        return upper_boundary, lower_boundary
    
    def _smooth_boundary(self, boundary, window_size=5):
        """
        使用移動平均平滑邊界
        """
        if len(boundary) < window_size:
            return boundary
            
        smoothed = boundary.copy()
        for i in range(window_size//2, len(boundary) - window_size//2):
            start_idx = i - window_size//2
            end_idx = i + window_size//2 + 1
            smoothed[i, 1] = np.mean(boundary[start_idx:end_idx, 1])
            
        return smoothed
    
    def calculate_distance_at_point(self, x):
        """
        計算指定x座標處的上下邊界距離
        """
        if self.upper_func is None or self.lower_func is None:
            return None
            
        if x < self.x_range[0] or x > self.x_range[1]:
            return None
            
        upper_y = self.upper_func(x)
        lower_y = self.lower_func(x)
        distance = abs(lower_y - upper_y)
        
        return {
            'x': x,
            'upper_y': upper_y,
            'lower_y': lower_y,
            'distance': distance
        }
    
    def get_all_distances(self, step=1):
        """
        獲取整個弧線範圍內的距離數據
        """
        if self.x_range is None:
            return None
            
        x_coords = np.arange(self.x_range[0], self.x_range[1] + 1, step)
        distances = []
        
        for x in x_coords:
            dist_info = self.calculate_distance_at_point(x)
            if dist_info:
                distances.append(dist_info)
                
        return distances
    
    def find_min_max_distance(self):
        """
        找到最小和最大距離點
        """
        distances = self.get_all_distances()
        if not distances:
            return None, None
            
        min_dist = min(distances, key=lambda x: x['distance'])
        max_dist = max(distances, key=lambda x: x['distance'])
        
        return min_dist, max_dist

def process_arc_image(model_path, image_path):
    """
    主要處理函數
    """
    # 載入模型和圖像
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.25, imgsz=1280)
    
    result = results[0]
    img = result.orig_img.copy()
    H, W = img.shape[:2]
    
    # 獲取分割遮罩
    mask = result.masks.data[0].cpu().numpy() if result.masks else None
    
    if mask is None:
        print("未檢測到分割遮罩")
        return img
    
    # 調整遮罩尺寸
    mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized > 0.5).astype(np.uint8)
    
    # 找輪廓
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未找到輪廓")
        return img
    
    # 選擇最大輪廓
    main_contour = max(contours, key=cv2.contourArea)
    
    # 創建邊界分析器
    analyzer = ArcBoundaryAnalyzer()
    upper_boundary, lower_boundary = analyzer.extract_arc_boundaries(main_contour, (H, W))
    
    if upper_boundary is None or lower_boundary is None:
        print("無法提取邊界")
        return img
    
    # 繪製原始輪廓 (灰色)
    cv2.drawContours(img, [main_contour], -1, (128, 128, 128), 1)
    
    # 繪製上邊界 (紅色)
    for i in range(len(upper_boundary) - 1):
        pt1 = tuple(upper_boundary[i].astype(int))
        pt2 = tuple(upper_boundary[i + 1].astype(int))
        cv2.line(img, pt1, pt2, (0, 0, 255), 3)
    
    # 繪製下邊界 (綠色)
    for i in range(len(lower_boundary) - 1):
        pt1 = tuple(lower_boundary[i].astype(int))
        pt2 = tuple(lower_boundary[i + 1].astype(int))
        cv2.line(img, pt1, pt2, (0, 255, 0), 3)
    
    # 找到最小和最大距離點
    min_dist, max_dist = analyzer.find_min_max_distance()
    
    if min_dist and max_dist:
        # 標記最小距離點 (藍色)
        min_x = int(min_dist['x'])
        min_upper_y = int(min_dist['upper_y'])
        min_lower_y = int(min_dist['lower_y'])
        
        cv2.circle(img, (min_x, min_upper_y), 6, (255, 0, 0), -1)
        cv2.circle(img, (min_x, min_lower_y), 6, (255, 0, 0), -1)
        cv2.line(img, (min_x, min_upper_y), (min_x, min_lower_y), (255, 0, 0), 2)
        cv2.putText(img, f'Min: {min_dist["distance"]:.1f}px', 
                   (min_x + 10, min_upper_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 標記最大距離點 (黃色)
        max_x = int(max_dist['x'])
        max_upper_y = int(max_dist['upper_y'])
        max_lower_y = int(max_dist['lower_y'])
        
        cv2.circle(img, (max_x, max_upper_y), 6, (0, 255, 255), -1)
        cv2.circle(img, (max_x, max_lower_y), 6, (0, 255, 255), -1)
        cv2.line(img, (max_x, max_upper_y), (max_x, max_lower_y), (0, 255, 255), 2)
        cv2.putText(img, f'Max: {max_dist["distance"]:.1f}px', 
                   (max_x + 10, max_lower_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 顯示統計信息
    distances = analyzer.get_all_distances()
    if distances:
        dist_values = [d['distance'] for d in distances]
        avg_distance = np.mean(dist_values)
        std_distance = np.std(dist_values)
        
        info_text = [
            f"Average Distance: {avg_distance:.2f} px",
            f"Std Deviation: {std_distance:.2f} px",
            f"Min Distance: {min_dist['distance']:.2f} px" if min_dist else "",
            f"Max Distance: {max_dist['distance']:.2f} px" if max_dist else "",
            f"Measurement Points: {len(distances)}"
        ]
        
        for i, text in enumerate(info_text):
            if text:
                cv2.putText(img, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 設置鼠標回調函數來即時顯示距離
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            dist_info = analyzer.calculate_distance_at_point(x)
            if dist_info:
                # 創建臨時圖像顯示當前點的距離
                temp_img = img.copy()
                
                # 繪製測量線
                upper_y = int(dist_info['upper_y'])
                lower_y = int(dist_info['lower_y'])
                
                cv2.line(temp_img, (x, upper_y), (x, lower_y), (255, 255, 255), 2)
                cv2.circle(temp_img, (x, upper_y), 4, (0, 0, 255), -1)
                cv2.circle(temp_img, (x, lower_y), 4, (0, 255, 0), -1)
                
                # 顯示距離
                cv2.putText(temp_img, f'Distance: {dist_info["distance"]:.1f}px', 
                           (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Arc Boundary Analysis - Interactive", temp_img)
    
    # 設置窗口和鼠標回調
    cv2.namedWindow("Arc Boundary Analysis - Interactive", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Arc Boundary Analysis - Interactive", mouse_callback)
    cv2.imshow("Arc Boundary Analysis - Interactive", img)
    
    print("移動鼠標查看任意點的距離測量")
    print("按任意鍵退出...")
    
    return img, analyzer

# 使用範例
if __name__ == "__main__":
    # 處理圖像
    result_img, analyzer = process_arc_image("yolov11-seg.pt", "your_image.jpg")
    
    # 也可以直接查詢特定點的距離
    # 例如查詢x=300處的距離
    x_query = 300
    dist_info = analyzer.calculate_distance_at_point(x_query)
    if dist_info:
        print(f"\n在x={x_query}處:")
        print(f"上邊界y座標: {dist_info['upper_y']:.2f}")
        print(f"下邊界y座標: {dist_info['lower_y']:.2f}")
        print(f"距離: {dist_info['distance']:.2f} 像素")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()