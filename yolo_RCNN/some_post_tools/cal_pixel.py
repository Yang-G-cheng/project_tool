import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import tkinter as tk
from tkinter import filedialog, messagebox

class ContourDistanceCalculator:
    def __init__(self):
        self.image = None
        self.gray_image = None
        self.pixel_to_mm = 0.1
        self.sensitivity = 50
        self.fig = None
        self.ax = None
        self.click_results = []
        
    def load_image(self, image_path=None):
        """仔入圖片"""
        if image_path is None:
            # 使用tkinter文件選圖片
            root = tk.Tk()
            root.withdraw()  # 隱藏主視窗
            image_path = filedialog.askopenfilename(
                title="選擇圖片文件",
                filetypes=[("圖片文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            )
            root.destroy()
            
        if not image_path:
            return False
            
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                messagebox.showerror("錯誤", "無法載入圖片文件")
                return False
                
            # 轉為RGB for matplot show
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            return True
        except Exception as e:
            messagebox.showerror("錯誤", f"載入圖片錯誤: {str(e)}")
            return False
    
    def get_brightness(self, x, y):
        """獲得指定位置亮度直"""
        if 0 <= x < self.gray_image.shape[1] and 0 <= y < self.gray_image.shape[0]:
            return int(self.gray_image[y, x])
        return 0
    
    def is_edge_point(self, x, y):
        """檢查指定點是否為邊界"""
        if x < 1 or x >= self.gray_image.shape[1] - 1 or y < 1 or y >= self.gray_image.shape[0] - 1:
            return False
            
        threshold = (100 - self.sensitivity) * 2.55
        
        # 獲得當前點與鄰點亮度
        current = self.get_brightness(x, y)
        neighbors = [
            self.get_brightness(x - 1, y),
            self.get_brightness(x + 1, y),
            self.get_brightness(x, y - 1),
            self.get_brightness(x, y + 1)
        ]
        
        # 檢查亮度差異
        for neighbor in neighbors:
            if abs(current - neighbor) > threshold:
                return True
        return False
    
    def find_nearest_edge(self, center_x, center_y, direction):
        """找最近點"""
        max_radius = 50
        
        for radius in range(1, max_radius + 1):
            for angle in range(0, 360, 10):
                rad = np.radians(angle)
                x = int(center_x + radius * np.cos(rad))
                y = int(center_y + radius * np.sin(rad) * direction)
                
                if (0 <= x < self.gray_image.shape[1] and 
                    0 <= y < self.gray_image.shape[0]):
                    if self.is_edge_point(x, y):
                        return y
        return None
    
    def calculate_distance(self, click_x, click_y):
        """計算指定點上下邊界距離"""
        if self.gray_image is None:
            return None
            
        # 找上邊界
        top_y = None
        for y in range(click_y, -1, -1):
            if self.is_edge_point(click_x, y):
                top_y = y
                break
        
        # 找下邊界
        bottom_y = None
        for y in range(click_y, self.gray_image.shape[0]):
            if self.is_edge_point(click_x, y):
                bottom_y = y
                break
        
        # 若找不到邊界 試著搜尋區域
        if top_y is None:
            top_y = self.find_nearest_edge(click_x, click_y, -1)
        if bottom_y is None:
            bottom_y = self.find_nearest_edge(click_x, click_y, 1)
        
        # 計算距離
        distance_pixels = 0
        if top_y is not None and bottom_y is not None:
            distance_pixels = abs(bottom_y - top_y)
        elif top_y is not None:
            distance_pixels = abs(click_y - top_y)
        elif bottom_y is not None:
            distance_pixels = abs(bottom_y - click_y)
        
        distance_mm = distance_pixels * self.pixel_to_mm
        
        return {
            'distance_mm': distance_mm,
            'distance_pixels': distance_pixels,
            'click_x': click_x,
            'click_y': click_y,
            'top_y': top_y,
            'bottom_y': bottom_y
        }
    
    def on_click(self, event):
        """滑鼠事件"""
        if event.inaxes != self.ax or self.image is None:
            return
            
        click_x = int(event.xdata)
        click_y = int(event.ydata)
        
        result = self.calculate_distance(click_x, click_y)
        if result:
            self.click_results.append(result)
            self.update_display(result)
            self.print_result(result)
    
    def update_display(self, result):
        """更新顯示 標記滑鼠點"""
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(f'輪廓邊界距離計算器 (像素轉換: 1px = {self.pixel_to_mm}mm)')
        
        # 歷史結果們
        for i, res in enumerate(self.click_results):
            # 點擊點 (红色)
            self.ax.plot(res['click_x'], res['click_y'], 'ro', markersize=8, 
                        label=f'點擊點 {i+1}' if i == 0 else "")
            
            # 上邊界 (藍)
            if res['top_y'] is not None:
                self.ax.plot(res['click_x'], res['top_y'], 'bo', markersize=6,
                           label=f'上邊界' if i == 0 else "")
            
            # 下邊界 (綠)
            if res['bottom_y'] is not None:
                self.ax.plot(res['click_x'], res['bottom_y'], 'go', markersize=6,
                           label=f'下邊界' if i == 0 else "")
            
            # 連線 (橘)
            if res['top_y'] is not None and res['bottom_y'] is not None:
                self.ax.plot([res['click_x'], res['click_x']], 
                           [res['top_y'], res['bottom_y']], 
                           'orange', linewidth=2, alpha=0.7)
            
            # 距離直
            text_y = res['click_y'] - 20 if res['click_y'] > 20 else res['click_y'] + 20
            self.ax.text(res['click_x'], text_y, 
                        f'{res["distance_mm"]:.2f}mm', 
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        fontsize=10, fontweight='bold')
        
        if len(self.click_results) > 0:
            self.ax.legend()
        
        plt.draw()
    
    def print_result(self, result):
        """顯示計算結果"""
        print(f"\n=== 距離計算結果 ===")
        print(f"點擊位置: ({result['click_x']}, {result['click_y']})")
        print(f"上邊界位置: {result['top_y'] if result['top_y'] is not None else '未找到'}")
        print(f"下邊界位置: {result['bottom_y'] if result['bottom_y'] is not None else '未找到'}")
        print(f"距離: {result['distance_pixels']} 像素 = {result['distance_mm']:.3f} 毫米")
        print("=" * 25)
    
    def update_sensitivity(self, val):
        """更新敏感度"""
        self.sensitivity = val
        print(f"邊界檢測敏感度已更新: {self.sensitivity}")
    
    def update_pixel_ratio(self, text):
        """更新像素比例"""
        try:
            self.pixel_to_mm = float(text)
            print(f"像素轉換比例已更新: 1 像素 = {self.pixel_to_mm} 毫米")
            # 重新计算所有结果
            for result in self.click_results:
                result['distance_mm'] = result['distance_pixels'] * self.pixel_to_mm
            if self.click_results:
                self.update_display(self.click_results[-1])
        except ValueError:
            print("無效像素比例值")
    
    def clear_results(self):
        """清除所有结果"""
        self.click_results = []
        if self.ax and self.image is not None:
            self.ax.clear()
            self.ax.imshow(self.image)
            self.ax.set_title(f'輪廓邊界計算器 (像素轉換: 1px = {self.pixel_to_mm}mm)')
            plt.draw()
        print("已清除所有結果")
    
    def run(self):
        """主程式"""
        print("輪廓邊界計算器")
        print("=" * 30)
        
        # 載入圖片
        if not self.load_image():
            print("無圖, 退出")
            return
        
        # 建立matplotlib界面
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle('輪廓邊界計算器', fontsize=16, fontweight='bold')
        
        # 調整子圖位置
        plt.subplots_adjust(bottom=0.25)
        
        # 顯示圖片
        self.ax.imshow(self.image)
        self.ax.set_title(f'點及圖像上任意位置計算距離 (像素轉換: 1px = {self.pixel_to_mm}mm)')
        
        # 敏感度滑竿
        ax_sensitivity = plt.axes([0.2, 0.1, 0.5, 0.03])
        slider_sensitivity = Slider(ax_sensitivity, '邊界檢測敏感度', 10, 100, 
                                  valinit=self.sensitivity, valfmt='%d')
        slider_sensitivity.on_changed(self.update_sensitivity)
        
        # 像素比例輸入框
        ax_ratio = plt.axes([0.2, 0.05, 0.15, 0.04])
        textbox_ratio = TextBox(ax_ratio, '像素/毫米: ', initial=str(self.pixel_to_mm))
        textbox_ratio.on_submit(self.update_pixel_ratio)
        
        # 點擊事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 鍵盤事件
        def on_key(event):
            if event.key == 'c':
                self.clear_results()
            elif event.key == 'q':
                plt.close()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        print("\n使用說明:")
        print("- 點擊影像上的任意位置計算到上下邊界的距離")
        print("- 調整敏感度滑桿改變邊界偵測敏感度")
        print("- 修改像素/毫米比例來調整單位轉換")
        print("- 按 'c' 鍵清除所有結果")
        print("- 按 'q' 鍵退出程式")
        print("- 關閉視窗退出程式")
        
        plt.show()

def main():
    """主函數"""
    try:
        calculator = ContourDistanceCalculator()
        calculator.run()
    except Exception as e:
        print(f"程式錯誤: {str(e)}")
        messagebox.showerror("錯誤", f"程式錯誤: {str(e)}")

if __name__ == "__main__":
    main()

# 使用示例:
# 1. 直接运行脚本会弹出文件选择对话框
# 2. 或者创建实例并指定图像路径:
#    calculator = ContourDistanceCalculator()
#    calculator.load_image("your_image_path.jpg")
#    calculator.run()

# 所需依赖包:
# pip install opencv-python matplotlib numpy tkinter