import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_wafer_boundaries(mask, method='contour'):
    """
    從YOLO11分割遮罩中提取wafer的上下邊界
    
    Args:
        mask: 二值化分割遮罩 (numpy array)
        method: 提取方法 ('contour', 'projection', 'fit_circle')
    
    Returns:
        上邊界點列表, 下邊界點列表
    """
    
    if method == 'contour':
        return extract_boundaries_contour(mask)
    elif method == 'projection':
        return extract_boundaries_projection(mask)
    elif method == 'fit_circle':
        return extract_boundaries_circle_fit(mask)

def extract_boundaries_contour(mask):
    """使用輪廓檢測方法提取邊界"""
    # 確保遮罩是二值化的
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 找到輪廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], []
    
    # 找到最大輪廓（應該是wafer邊緣）
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 提取輪廓點
    contour_points = largest_contour.reshape(-1, 2)
    
    # 分離上下邊界
    # 找到最高點和最低點的Y座標
    min_y = np.min(contour_points[:, 1])
    max_y = np.max(contour_points[:, 1])
    center_y = (min_y + max_y) / 2
    
    # 上邊界：Y座標小於中心點的點
    upper_boundary = contour_points[contour_points[:, 1] <= center_y]
    # 下邊界：Y座標大於中心點的點  
    lower_boundary = contour_points[contour_points[:, 1] >= center_y]
    
    # 按X座標排序
    upper_boundary = upper_boundary[np.argsort(upper_boundary[:, 0])]
    lower_boundary = lower_boundary[np.argsort(lower_boundary[:, 0])]
    
    return upper_boundary, lower_boundary

def extract_boundaries_projection(mask):
    """使用投影方法提取邊界"""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    height, width = binary_mask.shape
    upper_boundary = []
    lower_boundary = []
    
    # 對每一列(X座標)進行垂直投影
    for x in range(width):
        column = binary_mask[:, x]
        white_pixels = np.where(column == 255)[0]
        
        if len(white_pixels) > 0:
            # 上邊界是該列第一個白色像素
            upper_y = white_pixels[0]
            # 下邊界是該列最後一個白色像素
            lower_y = white_pixels[-1]
            
            upper_boundary.append([x, upper_y])
            lower_boundary.append([x, lower_y])
    
    return np.array(upper_boundary), np.array(lower_boundary)

def extract_boundaries_circle_fit(mask):
    """使用圓形擬合方法提取邊界"""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 找到輪廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], []
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 擬合圓形
    (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
    
    # 生成圓形邊界點
    angles = np.linspace(0, 2*np.pi, 360)
    circle_points = []
    
    for angle in angles:
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        circle_points.append([x, y])
    
    circle_points = np.array(circle_points)
    
    # 分離上下半圓
    upper_boundary = circle_points[circle_points[:, 1] <= center_y]
    lower_boundary = circle_points[circle_points[:, 1] >= center_y]
    
    # 按X座標排序
    upper_boundary = upper_boundary[np.argsort(upper_boundary[:, 0])]
    lower_boundary = lower_boundary[np.argsort(lower_boundary[:, 0])]
    
    return upper_boundary, lower_boundary

def visualize_boundaries(image, upper_boundary, lower_boundary, title="Wafer Boundaries"):
    """視覺化邊界結果"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap='gray')
    
    if len(upper_boundary) > 0:
        plt.plot(upper_boundary[:, 0], upper_boundary[:, 1], 'r-', linewidth=2, label='Upper Boundary')
    
    if len(lower_boundary) > 0:
        plt.plot(lower_boundary[:, 0], lower_boundary[:, 1], 'b-', linewidth=2, label='Lower Boundary')
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def get_boundary_equations(upper_boundary, lower_boundary):
    """獲得邊界的數學方程式（多項式擬合）"""
    results = {}
    
    if len(upper_boundary) > 0:
        # 對上邊界進行多項式擬合
        upper_poly = np.polyfit(upper_boundary[:, 0], upper_boundary[:, 1], deg=2)
        results['upper_equation'] = upper_poly
        results['upper_function'] = np.poly1d(upper_poly)
    
    if len(lower_boundary) > 0:
        # 對下邊界進行多項式擬合
        lower_poly = np.polyfit(lower_boundary[:, 0], lower_boundary[:, 1], deg=2)
        results['lower_equation'] = lower_poly
        results['lower_function'] = np.poly1d(lower_poly)
    
    return results

# 使用範例
def main():
    # 假設你有YOLO11的分割遮罩
    # mask = your_yolo_segmentation_mask
    
    # 方法1: 使用輪廓檢測
    upper_boundary, lower_boundary = extract_wafer_boundaries(mask, method='contour')
    
    # 方法2: 使用投影方法
    # upper_boundary, lower_boundary = extract_wafer_boundaries(mask, method='projection')
    
    # 方法3: 使用圓形擬合
    # upper_boundary, lower_boundary = extract_wafer_boundaries(mask, method='fit_circle')
    
    # 視覺化結果
    # visualize_boundaries(original_image, upper_boundary, lower_boundary)
    
    # 獲得邊界方程式
    boundary_equations = get_boundary_equations(upper_boundary, lower_boundary)
    
    print("Upper boundary points:", len(upper_boundary))
    print("Lower boundary points:", len(lower_boundary))
    
    if 'upper_function' in boundary_equations:
        print("Upper boundary equation:", boundary_equations['upper_equation'])
    
    if 'lower_function' in boundary_equations:
        print("Lower boundary equation:", boundary_equations['lower_equation'])
    
    return upper_boundary, lower_boundary, boundary_equations

# 如果你想要計算特定X座標的Y值
def get_y_at_x(boundary_function, x_value):
    """根據邊界函數計算特定X座標的Y值"""
    return boundary_function(x_value)

if __name__ == "__main__":
    # main()
    print("Wafer boundary extraction functions ready to use!")
    # 1. 載入你的YOLO11分割結果遮罩
    mask = your_yolo_segmentation_mask  # 應該是二值化遮罩

    # 2. 提取邊界
    upper_boundary, lower_boundary = extract_wafer_boundaries(mask, method='contour')

    # 3. 視覺化結果
    visualize_boundaries(original_image, upper_boundary, lower_boundary)

    # 4. 獲得數學方程式
    boundary_equations = get_boundary_equations(upper_boundary, lower_boundary)

    # 5. 使用邊界函數計算特定位置的Y值
    if 'upper_function' in boundary_equations:
        y_upper = boundary_equations['upper_function'](x_coordinate)
        
    if 'lower_function' in boundary_equations:
        y_lower = boundary_equations['lower_function'](x_coordinate)