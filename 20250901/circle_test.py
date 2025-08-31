import cv2
import numpy as np
import os
import math

def find_and_extract_arc_rectangles(image_path, rect_size=(20, 20), angle_step=5):
    """
    自動偵測圖片中的圓，並沿著圓弧提取矩形區域，轉正後儲存。
    """
    # --- 1. 讀取圖片 ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：無法讀取圖片於 '{image_path}'")
        return

    img_display = img.copy()
    output_dir = "extracted_rectangles"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 創建5x5裁切圖片的儲存資料夾
    output_dir_5x5 = "extracted_rectangles_5x5"
    if not os.path.exists(output_dir_5x5):
        os.makedirs(output_dir_5x5)

    # --- 2. 自動偵測圓 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=gray.shape[0] / 4,
        param1=100,
        param2=30,
        minRadius=int(gray.shape[1] / 6),
        maxRadius=int(gray.shape[1] / 2)
    )

    if circles is None:
        print("錯誤：在圖片中沒有偵測到圓。")
        cv2.imshow("No circles found", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    circles = np.uint16(np.around(circles))
    circle = circles[0, 0]
    center = (circle[0], circle[1])
    radius = circle[2]
    print(f"偵測到圓: 中心點={center}, 半徑={radius}")

    cv2.circle(img_display, center, radius, (0, 0, 255), 3)
    cv2.circle(img_display, center, 5, (255, 0, 255), -1)

    # --- 3. 沿著圓弧提取、轉正並儲存矩形 ---
    rect_width, rect_height = rect_size
    count = 0
    
    for angle in np.arange(0, 360, angle_step):
        rad = math.radians(angle)
        x = center[0] + radius * math.cos(rad)
        y = center[1] + radius * math.sin(rad)

        rotation_angle_deg = angle + 90
        rect_on_arc = (
            (float(x), float(y)),
            (float(rect_width), float(rect_height)),
            float(rotation_angle_deg)
        )
        src_pts = cv2.boxPoints(rect_on_arc).astype("float32")

        dst_pts = np.array([
            [0, rect_height - 1],
            [0, 0],
            [rect_width - 1, 0],
            [rect_width - 1, rect_height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        img_with_line = img.copy()
        end_point_x = int(center[0] + radius * 5 * math.cos(rad))
        end_point_y = int(center[1] + radius * 5 * math.sin(rad))
        line_color = (0, 0, 255)
        line_thickness = 1
        cv2.line(img_with_line, (int(center[0]), int(center[1])), (end_point_x, end_point_y), line_color, line_thickness)

        warped = cv2.warpPerspective(img_with_line, M, (rect_width, rect_height))

        output_path = os.path.join(output_dir, f"rect_{count:03d}_angle_{angle}.png")
        cv2.imwrite(output_path, warped)

        # --- 以中心線為基準，裁切5x5的圖片 ---
        crop_size = 5
        center_x = rect_width // 2
        center_y = rect_height // 2
        x1 = center_x - (crop_size // 2)
        y1 = center_y - (crop_size // 2)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        if x1 >= 0 and y1 >= 0 and x2 <= warped.shape[1] and y2 <= warped.shape[0]:
            cropped_5x5 = warped[y1:y2, x1:x2]
            output_path_5x5 = os.path.join(output_dir_5x5, f"rect_5x5_{count:03d}_angle_{angle}.png")
            cv2.imwrite(output_path_5x5, cropped_5x5)

        count += 1
        cv2.drawContours(img_display, [np.intp(src_pts)], 0, (0, 255, 0), 2)

    print(f"成功提取並儲存了 {count} 個 20x20 矩形到 '{output_dir}' 資料夾")
    print(f"同時儲存了 {count} 個 5x5 矩形到 '{output_dir_5x5}' 資料夾")

    # --- 4. 顯示最終結果 ---
    cv2.imshow("Detected Circle and Extracted Rectangles", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\YAN\use_gemini\circle_test\your_image.jpg"
    find_and_extract_arc_rectangles(image_path, rect_size=(20, 20))
