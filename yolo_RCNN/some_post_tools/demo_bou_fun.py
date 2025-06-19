import numpy as np
import matplotlib.pyplot as plt

# 模擬wafer邊界提取的範例
def demonstrate_boundary_function():
    """演示邊界函數的概念和使用"""
    
    # 1. 模擬從YOLO分割結果中提取的wafer邊界點
    print("=== 邊界函數使用範例 ===\n")
    
    # 假設我們從wafer分割結果中提取到這些邊界點
    upper_boundary_points = np.array([
        [100, 50], [150, 45], [200, 42], [250, 41], [300, 42], 
        [350, 45], [400, 50], [450, 57], [500, 65]
    ])
    
    lower_boundary_points = np.array([
        [100, 150], [150, 155], [200, 158], [250, 159], [300, 158], 
        [350, 155], [400, 150], [450, 143], [500, 135]
    ])
    
    print("提取到的上邊界點:")
    print(f"X座標: {upper_boundary_points[:, 0]}")
    print(f"Y座標: {upper_boundary_points[:, 1]}\n")
    
    # 2. 建立邊界函數（多項式擬合）
    print("=== 建立邊界函數 ===")
    
    # 對上邊界進行2次多項式擬合
    upper_poly_coeffs = np.polyfit(upper_boundary_points[:, 0], upper_boundary_points[:, 1], deg=2)
    upper_function = np.poly1d(upper_poly_coeffs)
    
    # 對下邊界進行2次多項式擬合
    lower_poly_coeffs = np.polyfit(lower_boundary_points[:, 0], lower_boundary_points[:, 1], deg=2)
    lower_function = np.poly1d(lower_poly_coeffs)
    
    print(f"上邊界函數係數: {upper_poly_coeffs}")
    print(f"上邊界函數: y = {upper_poly_coeffs[0]:.6f}x² + {upper_poly_coeffs[1]:.6f}x + {upper_poly_coeffs[2]:.6f}")
    print(f"下邊界函數係數: {lower_poly_coeffs}")
    print(f"下邊界函數: y = {lower_poly_coeffs[0]:.6f}x² + {lower_poly_coeffs[1]:.6f}x + {lower_poly_coeffs[2]:.6f}\n")
    
    # 3. 計算特定位置的Y值
    print("=== 計算特定位置的Y值 ===")
    
    # 假設我們想知道X=275位置的上下邊界Y值
    target_x = 275
    upper_y = upper_function(target_x)
    lower_y = lower_function(target_x)
    
    print(f"當 X = {target_x} 時:")
    print(f"  上邊界 Y = {upper_y:.2f}")
    print(f"  下邊界 Y = {lower_y:.2f}")
    print(f"  wafer在此位置的高度 = {lower_y - upper_y:.2f} 像素\n")
    
    # 4. 計算多個位置的Y值
    print("=== 計算多個位置的Y值 ===")
    test_x_values = [200, 250, 300, 350, 400]
    
    print("X座標\t上邊界Y\t下邊界Y\tWafer高度")
    print("-" * 45)
    for x in test_x_values:
        upper_y = upper_function(x)
        lower_y = lower_function(x)
        height = lower_y - upper_y
        print(f"{x}\t{upper_y:.1f}\t{lower_y:.1f}\t{height:.1f}")
    
    # 5. 實際應用場景
    print("\n=== 實際應用場景 ===")
    print("這個功能的用途:")
    print("1. 品質檢測: 檢查wafer邊緣是否在允許範圍內")
    print("2. 缺陷檢測: 檢測邊緣是否有裂縫或缺口")
    print("3. 尺寸測量: 測量wafer在任意位置的寬度/高度")
    print("4. 座標轉換: 將相對位置轉換為絕對座標")
    print("5. 路徑規劃: 為機械臂或檢測設備規劃路徑")
    
    return upper_function, lower_function, upper_boundary_points, lower_boundary_points

def practical_applications(upper_function, lower_function):
    """展示實際應用場景"""
    print("\n=== 實際應用範例 ===")
    
    # 應用1: 品質檢測
    print("1. 品質檢測範例:")
    tolerance = 5  # 允許誤差5像素
    check_points = [200, 300, 400]
    
    for x in check_points:
        upper_y = upper_function(x)
        expected_upper = 45  # 預期的上邊界位置
        deviation = abs(upper_y - expected_upper)
        
        status = "合格" if deviation <= tolerance else "不合格"
        print(f"  X={x}: 實際={upper_y:.1f}, 預期={expected_upper}, 偏差={deviation:.1f}, 狀態={status}")
    
    # 應用2: 尺寸測量
    print("\n2. 尺寸測量範例:")
    measure_x = 300
    upper_y = upper_function(measure_x)
    lower_y = lower_function(measure_x)
    wafer_height = lower_y - upper_y
    
    print(f"  在X={measure_x}位置:")
    print(f"  Wafer高度 = {wafer_height:.1f} 像素")
    print(f"  如果1像素=0.1mm，則實際高度 = {wafer_height*0.1:.1f} mm")
    
    # 應用3: 檢測路徑生成
    print("\n3. 檢測路徑生成:")
    scan_x_positions = np.linspace(150, 450, 10)  # 生成10個檢測點
    
    print("  檢測路徑座標:")
    for i, x in enumerate(scan_x_positions):
        center_y = (upper_function(x) + lower_function(x)) / 2  # wafer中心線
        print(f"    點{i+1}: ({x:.0f}, {center_y:.1f})")

def visualize_boundary_functions(upper_function, lower_function, upper_points, lower_points):
    """視覺化邊界函數"""
    plt.figure(figsize=(12, 8))
    
    # 生成平滑的函數曲線
    x_smooth = np.linspace(100, 500, 200)
    upper_smooth = upper_function(x_smooth)
    lower_smooth = lower_function(x_smooth)
    
    # 繪製原始邊界點
    plt.scatter(upper_points[:, 0], upper_points[:, 1], color='red', s=50, label='上邊界點', zorder=5)
    plt.scatter(lower_points[:, 0], lower_points[:, 1], color='blue', s=50, label='下邊界點', zorder=5)
    
    # 繪製擬合的函數曲線
    plt.plot(x_smooth, upper_smooth, 'r-', linewidth=2, label='上邊界函數', alpha=0.8)
    plt.plot(x_smooth, lower_smooth, 'b-', linewidth=2, label='下邊界函數', alpha=0.8)
    
    # 標示特定位置的計算結果
    test_x = 275
    test_upper_y = upper_function(test_x)
    test_lower_y = lower_function(test_x)
    
    plt.axvline(x=test_x, color='green', linestyle='--', alpha=0.7, label=f'測試位置 X={test_x}')
    plt.plot(test_x, test_upper_y, 'go', markersize=8, label=f'上邊界Y={test_upper_y:.1f}')
    plt.plot(test_x, test_lower_y, 'go', markersize=8, label=f'下邊界Y={test_lower_y:.1f}')
    
    plt.xlabel('X座標 (像素)')
    plt.ylabel('Y座標 (像素)')
    plt.title('Wafer邊界函數示意圖')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # 反轉Y軸，符合圖像座標系
    
    plt.tight_layout()
    plt.show()

# 執行演示
if __name__ == "__main__":
    # 執行主要演示
    upper_func, lower_func, upper_pts, lower_pts = demonstrate_boundary_function()
    
    # 展示實際應用
    practical_applications(upper_func, lower_func)
    
    # 視覺化結果
    # visualize_boundary_functions(upper_func, lower_func, upper_pts, lower_pts)