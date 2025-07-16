import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import math

class WaferEdgeAnalyzer:
    def __init__(self):
        self.mask = None
        self.mask_coords = None
        self.cleaned_coords = None
        self.inner_boundary = None
        self.outer_boundary = None
        self.center = None
        self.radius = None
        self.image_shape = None
        
    def load_yolo_mask(self, mask):
        """
        載入YOLO11預測的mask
        
        Args:
            mask: YOLO11預測的mask，可以是：
                - 二值化numpy array (H, W)
                - 布林numpy array (H, W)
                - 0和1的numpy array (H, W)
        """
        if isinstance(mask, list):
            mask = np.array(mask)
        
        # 確保mask是二值化的
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        
        self.mask = mask
        self.image_shape = mask.shape
        
        # 從mask中提取邊緣座標
        self.extract_coordinates_from_mask()
        
        print(f"載入mask尺寸: {self.image_shape}")
        print(f"提取到 {len(self.mask_coords)} 個邊緣座標點")
        
        return self.mask
    
    def extract_coordinates_from_mask(self):
        """從mask中提取邊緣座標"""
        if self.mask is None:
            raise ValueError("請先載入mask")
        
        # 找到mask的輪廓
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            raise ValueError("未在mask中找到輪廓")
        
        # 選擇最大的輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 提取座標點
        self.mask_coords = largest_contour.reshape(-1, 2)
        
        return self.mask_coords
    
    def detect_and_remove_outliers(self, method='dbscan', eps=5, min_samples=10, 
                                  z_threshold=3, distance_threshold=30):
        """
        檢測並移除離群座標
        
        Args:
            method: 'dbscan', 'z_score', 'distance', 'morphology' 中的一種
            eps: DBSCAN的鄰域半徑
            min_samples: DBSCAN的最小樣本數
            z_threshold: Z-score閾值
            distance_threshold: 距離閾值
        """
        if self.mask_coords is None:
            raise ValueError("請先載入mask座標")
        
        coords = self.mask_coords.copy()
        original_count = len(coords)
        
        if method == 'dbscan':
            # 使用DBSCAN聚類檢測離群點
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            labels = clustering.labels_
            
            # 保留最大聚類的點
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels if label != -1]
                if cluster_sizes:
                    largest_cluster = max(cluster_sizes, key=lambda x: x[1])[0]
                    coords = coords[labels == largest_cluster]
            
        elif method == 'z_score':
            # 使用Z-score檢測離群點
            center_x, center_y = np.mean(coords, axis=0)
            distances = np.sqrt((coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2)
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            if std_dist > 0:
                z_scores = np.abs((distances - mean_dist) / std_dist)
                coords = coords[z_scores < z_threshold]
            
        elif method == 'distance':
            # 使用距離閾值檢測離群點
            dist_matrix = cdist(coords, coords)
            avg_distances = np.mean(dist_matrix, axis=1)
            coords = coords[avg_distances < distance_threshold]
            
        elif method == 'morphology':
            # 使用形態學操作清理mask
            kernel = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
            
            # 重新提取座標
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                coords = largest_contour.reshape(-1, 2)
        
        self.cleaned_coords = coords
        removed_count = original_count - len(coords)
        print(f"移除了 {removed_count} 個離群座標點，剩餘 {len(coords)} 個點")
        
        return self.cleaned_coords
    
    def smooth_coordinates(self, window_size=5):
        """
        對座標進行平滑處理
        
        Args:
            window_size: 平滑窗口大小
        """
        if self.cleaned_coords is None:
            raise ValueError("請先執行離群點檢測")
        
        coords = self.cleaned_coords.copy()
        
        # 按角度排序座標
        center_x, center_y = np.mean(coords, axis=0)
        angles = np.arctan2(coords[:, 1] - center_y, coords[:, 0] - center_x)
        sorted_indices = np.argsort(angles)
        sorted_coords = coords[sorted_indices]
        
        # 使用移動平均進行平滑
        def moving_average(data, window):
            if len(data) < window:
                return data
            
            # 擴展數據以處理邊界
            extended_data = np.concatenate([data[-window//2:], data, data[:window//2]])
            
            smoothed = np.convolve(extended_data, np.ones(window)/window, mode='valid')
            return smoothed
        
        # 分別對x和y座標進行平滑
        smooth_x = moving_average(sorted_coords[:, 0], window_size)
        smooth_y = moving_average(sorted_coords[:, 1], window_size)
        
        self.cleaned_coords = np.column_stack([smooth_x, smooth_y]).astype(int)
        
        return self.cleaned_coords
    
    def find_wafer_center_and_radius(self):
        """找到晶圓中心和半徑"""
        if self.cleaned_coords is None:
            raise ValueError("請先執行座標清理")
        
        coords = self.cleaned_coords
        
        # 使用最小二乘法擬合圓形
        def fit_circle(points):
            x = points[:, 0]
            y = points[:, 1]
            
            # 設置方程組 (x-a)² + (y-b)² = r²
            # 展開為 x² + y² - 2ax - 2by + (a² + b² - r²) = 0
            A = np.column_stack([2*x, 2*y, np.ones(len(x))])
            b = x**2 + y**2
            
            try:
                params = np.linalg.lstsq(A, b, rcond=None)[0]
                center_x, center_y, c = params
                radius = np.sqrt(center_x**2 + center_y**2 + c)
                return (int(center_x), int(center_y)), int(radius)
            except:
                # 如果擬合失敗，使用質心和平均距離
                center_x, center_y = np.mean(coords, axis=0)
                distances = np.sqrt((coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2)
                radius = np.mean(distances)
                return (int(center_x), int(center_y)), int(radius)
        
        self.center, self.radius = fit_circle(coords)
        print(f"晶圓中心: {self.center}, 半徑: {self.radius}")
        
        return self.center, self.radius
    
    def extract_inner_outer_boundaries(self, thickness_pixels=10):
        """
        從mask邊緣提取內外邊界
        
        Args:
            thickness_pixels: 邊緣厚度（像素）
        """
        if self.center is None or self.radius is None:
            raise ValueError("請先找到晶圓中心和半徑")
        
        # 使用形態學操作生成內外邊界
        kernel = np.ones((thickness_pixels, thickness_pixels), np.uint8)
        
        # 外邊界：原始mask的外輪廓
        outer_mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        outer_contours, _ = cv2.findContours(outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 內邊界：腐蝕後的mask輪廓
        inner_mask = cv2.erode(self.mask, kernel, iterations=1)
        inner_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 選擇最大輪廓
        if outer_contours:
            largest_outer = max(outer_contours, key=cv2.contourArea)
            self.outer_boundary = largest_outer.reshape(-1, 2)
        else:
            self.outer_boundary = self.cleaned_coords
        
        if inner_contours:
            largest_inner = max(inner_contours, key=cv2.contourArea)
            self.inner_boundary = largest_inner.reshape(-1, 2)
        else:
            # 如果沒有內輪廓，創建一個稍小的圓形邊界
            center_x, center_y = self.center
            inner_radius = self.radius - thickness_pixels
            angles = np.linspace(0, 2*np.pi, 200)
            inner_x = center_x + inner_radius * np.cos(angles)
            inner_y = center_y + inner_radius * np.sin(angles)
            self.inner_boundary = np.column_stack([inner_x, inner_y]).astype(int)
        
        print(f"外邊界點數: {len(self.outer_boundary)}")
        print(f"內邊界點數: {len(self.inner_boundary)}")
        
        return self.inner_boundary, self.outer_boundary
    
    def get_76_points_on_boundary(self, boundary_points):
        """在邊界上獲取76個等間距點"""
        if boundary_points is None or len(boundary_points) == 0:
            return []
        
        center_x, center_y = self.center
        
        # 計算每個點到中心的角度
        angles = np.arctan2(boundary_points[:, 1] - center_y, boundary_points[:, 0] - center_x)
        
        # 按角度排序
        sorted_indices = np.argsort(angles)
        sorted_points = boundary_points[sorted_indices]
        sorted_angles = angles[sorted_indices]
        
        # 確保角度範圍覆蓋完整圓周
        if len(sorted_angles) > 1:
            angle_range = sorted_angles[-1] - sorted_angles[0]
            if angle_range < 5:  # 如果角度範圍太小，擴展到完整圓周
                # 添加週期性邊界
                sorted_angles = np.append(sorted_angles, sorted_angles[0] + 2*np.pi)
                sorted_points = np.append(sorted_points, [sorted_points[0]], axis=0)
        
        # 創建插值函數
        if len(sorted_points) > 3:
            f_x = interp1d(sorted_angles, sorted_points[:, 0], kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
            f_y = interp1d(sorted_angles, sorted_points[:, 1], kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
            
            # 生成76個等間距角度
            target_angles = np.linspace(sorted_angles[0], sorted_angles[-1], 76)
            target_x = f_x(target_angles)
            target_y = f_y(target_angles)
            
            selected_points = np.column_stack([target_x, target_y]).astype(int)
        else:
            # 如果點太少，直接複製現有點
            selected_points = np.tile(sorted_points, (76//len(sorted_points)+1, 1))[:76]
        
        return selected_points
    
    def calculate_distances(self, inner_points, outer_points):
        """計算對應點之間的距離"""
        if len(inner_points) != len(outer_points):
            min_len = min(len(inner_points), len(outer_points))
            inner_points = inner_points[:min_len]
            outer_points = outer_points[:min_len]
        
        distances = []
        for i in range(len(inner_points)):
            dist = np.sqrt((inner_points[i][0] - outer_points[i][0])**2 + 
                          (inner_points[i][1] - outer_points[i][1])**2)
            distances.append(dist)
        
        return distances
    
    def visualize_results(self, inner_points, outer_points, distances):
        """可視化結果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 顯示原始mask
        ax1.imshow(self.mask, cmap='gray')
        ax1.set_title('原始YOLO Mask')
        ax1.axis('off')
        
        # 2. 顯示座標清理結果
        if self.mask_coords is not None:
            ax2.scatter(self.mask_coords[:, 0], self.mask_coords[:, 1], 
                       c='red', s=1, alpha=0.5, label='原始邊緣座標')
        if self.cleaned_coords is not None:
            ax2.scatter(self.cleaned_coords[:, 0], self.cleaned_coords[:, 1], 
                       c='blue', s=1, alpha=0.7, label='清理後座標')
        
        # 顯示內外邊界
        if self.inner_boundary is not None:
            ax2.scatter(self.inner_boundary[:, 0], self.inner_boundary[:, 1], 
                       c='green', s=1, alpha=0.5, label='內邊界')
        if self.outer_boundary is not None:
            ax2.scatter(self.outer_boundary[:, 0], self.outer_boundary[:, 1], 
                       c='orange', s=1, alpha=0.5, label='外邊界')
        
        ax2.set_title('邊緣座標處理結果')
        ax2.legend()
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        
        # 3. 顯示76個對應點和連接線
        ax3.imshow(self.mask, cmap='gray', alpha=0.3)
        
        # 繪製76個點
        for i, point in enumerate(inner_points):
            ax3.plot(point[0], point[1], 'go', markersize=4)
        for i, point in enumerate(outer_points):
            ax3.plot(point[0], point[1], 'ro', markersize=4)
        
        # 繪製連接線
        for i in range(len(inner_points)):
            ax3.plot([inner_points[i][0], outer_points[i][0]], 
                    [inner_points[i][1], outer_points[i][1]], 'b-', linewidth=1, alpha=0.7)
        
        # 標記中心點
        if self.center:
            ax3.plot(self.center[0], self.center[1], 'k+', markersize=10, markeredgewidth=2)
        
        ax3.set_title('76個對應點分析結果\n綠色：內邊界，紅色：外邊界，藍線：對應關係')
        ax3.axis('off')
        
        # 4. 顯示距離分布
        ax4.plot(range(len(distances)), distances, 'b-', linewidth=2, marker='o', markersize=3)
        ax4.set_title(f'內外邊界距離分布 (共{len(distances)}個點)')
        ax4.set_xlabel('點編號')
        ax4.set_ylabel('距離 (像素)')
        ax4.grid(True, alpha=0.3)
        
        # 添加統計信息
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        ax4.axhline(y=mean_dist, color='r', linestyle='--', alpha=0.7, label=f'平均值: {mean_dist:.2f}')
        ax4.axhline(y=mean_dist + std_dist, color='orange', linestyle='--', alpha=0.7, 
                   label=f'平均值+標準差: {mean_dist + std_dist:.2f}')
        ax4.axhline(y=mean_dist - std_dist, color='orange', linestyle='--', alpha=0.7, 
                   label=f'平均值-標準差: {mean_dist - std_dist:.2f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    def analyze_wafer_from_mask(self, mask, outlier_method='morphology', 
                               thickness_pixels=10, smooth_window=5):
        """
        從YOLO mask完整分析晶圓
        
        Args:
            mask: YOLO11預測的mask
            outlier_method: 離群點檢測方法
            thickness_pixels: 邊緣厚度
            smooth_window: 平滑窗口大小
        """
        print("=== 開始晶圓邊緣分析 ===")
        
        print("正在載入YOLO mask...")
        self.load_yolo_mask(mask)
        
        print("正在檢測並移除離群座標...")
        self.detect_and_remove_outliers(method=outlier_method)
        
        print("正在平滑座標...")
        self.smooth_coordinates(window_size=smooth_window)
        
        print("正在找到晶圓中心和半徑...")
        self.find_wafer_center_and_radius()
        
        print("正在提取內外邊界...")
        self.extract_inner_outer_boundaries(thickness_pixels=thickness_pixels)
        
        print("正在獲取76個等間距對應點...")
        inner_points = self.get_76_points_on_boundary(self.inner_boundary)
        outer_points = self.get_76_points_on_boundary(self.outer_boundary)
        
        print("正在計算距離...")
        distances = self.calculate_distances(inner_points, outer_points)
        
        print("正在可視化結果...")
        self.visualize_results(inner_points, outer_points, distances)
        
        # 統計信息
        print(f"\n=== 分析結果 ===")
        print(f"Mask尺寸: {self.image_shape}")
        print(f"原始邊緣座標點數: {len(self.mask_coords)}")
        print(f"清理後座標點數: {len(self.cleaned_coords)}")
        print(f"晶圓中心: {self.center}")
        print(f"晶圓半徑: {self.radius}")
        print(f"內邊界點數: {len(self.inner_boundary)}")
        print(f"外邊界點數: {len(self.outer_boundary)}")
        print(f"對應點數: {len(inner_points)}")
        print(f"平均距離: {np.mean(distances):.2f} 像素")
        print(f"最大距離: {np.max(distances):.2f} 像素")
        print(f"最小距離: {np.min(distances):.2f} 像素")
        print(f"距離標準差: {np.std(distances):.2f} 像素")
        
        return {
            'mask': self.mask,
            'original_coords': self.mask_coords,
            'cleaned_coords': self.cleaned_coords,
            'inner_boundary': self.inner_boundary,
            'outer_boundary': self.outer_boundary,
            'inner_points': inner_points,
            'outer_points': outer_points,
            'distances': distances,
            'center': self.center,
            'radius': self.radius
        }

# 使用範例
if __name__ == "__main__":
    # 創建分析器實例
    analyzer = WaferEdgeAnalyzer()
    
    # 示例：創建一個模擬的YOLO mask
    # 實際使用時，這應該是您從YOLO11得到的mask
    height, width = 800, 800
    center_x, center_y = width // 2, height // 2
    outer_radius = 300
    inner_radius = 280
    
    # 創建環形mask
    y, x = np.ogrid[:height, :width]
    outer_mask = (x - center_x)**2 + (y - center_y)**2 <= outer_radius**2
    inner_mask = (x - center_x)**2 + (y - center_y)**2 <= inner_radius**2
    ring_mask = (outer_mask & ~inner_mask).astype(np.uint8) * 255
    
    try:
        # 分析晶圓
        results = analyzer.analyze_wafer_from_mask(
            mask=ring_mask,
            outlier_method='morphology',
            thickness_pixels=10,
            smooth_window=5
        )
        
        # 保存結果到CSV
        import pandas as pd
        df = pd.DataFrame({
            'point_index': range(len(results['distances'])),
            'inner_x': [p[0] for p in results['inner_points']],
            'inner_y': [p[1] for p in results['inner_points']],
            'outer_x': [p[0] for p in results['outer_points']],
            'outer_y': [p[1] for p in results['outer_points']],
            'distance': results['distances']
        })
        df.to_csv('wafer_edge_analysis.csv', index=False)
        print("\n數據已保存到 wafer_edge_analysis.csv")
        
    except Exception as e:
        print(f"錯誤: {e}")
        print("請確保已安裝所需的庫:")
        print("pip install opencv-python numpy matplotlib scikit-learn pandas scipy")
