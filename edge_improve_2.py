import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------
# Version 1: 單筆圖像 RANSAC 邊緣擬合
# ---------------------------
def fit_circle_ransac(points, max_trials=1000, residual_threshold=2.0):
    def calc_circle(p1, p2, p3):
        A = np.array([[p1[0], p1[1], 1],
                      [p2[0], p2[1], 1],
                      [p3[0], p3[1], 1]])
        if np.linalg.matrix_rank(A) < 3:
            return None
        A1 = np.linalg.det([[p1[0]**2 + p1[1]**2, p1[1], 1],
                            [p2[0]**2 + p2[1]**2, p2[1], 1],
                            [p3[0]**2 + p3[1]**2, p3[1], 1]])
        A2 = np.linalg.det([[p1[0]**2 + p1[1]**2, p1[0], 1],
                            [p2[0]**2 + p2[1]**2, p2[0], 1],
                            [p3[0]**2 + p3[1]**2, p3[0], 1]])
        A3 = np.linalg.det([[p1[0]**2 + p1[1]**2, p1[0], p1[1]],
                            [p2[0]**2 + p2[1]**2, p2[0], p2[1]],
                            [p3[0]**2 + p3[1]**2, p3[0], p3[1]]])
        A4 = np.linalg.det(A)
        if A4 == 0:
            return None
        cx = 0.5 * A1 / A4
        cy = -0.5 * A2 / A4
        r = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        return cx, cy, r

    best_inliers = []
    best_circle = None
    for _ in range(max_trials):
        samples = points[np.random.choice(points.shape[0], 3, replace=False)]
        circle = calc_circle(*samples)
        if circle is None:
            continue
        cx, cy, r = circle
        dists = np.abs(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r)
        inliers = points[dists < residual_threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_circle = (cx, cy, r)
    return best_circle, best_inliers


def process_single_image(image_path):
    mask = cv2.imread(image_path, 0)
    edges = cv2.Canny(mask, 50, 150)
    ys, xs = np.where(edges > 0)
    points = np.stack((xs, ys), axis=-1)
    
    circle_ransac, inliers = fit_circle_ransac(points)
    overlay = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)

    if circle_ransac:
        cx, cy, r = circle_ransac
        theta = np.linspace(0, 2 * np.pi, 360)
        arc_x = (cx + r * np.cos(theta)).astype(np.int32)
        arc_y = (cy + r * np.sin(theta)).astype(np.int32)
        valid = (arc_x >= 0) & (arc_x < mask.shape[1]) & (arc_y >= 0) & (arc_y < mask.shape[0])
        overlay[arc_y[valid], arc_x[valid]] = [0, 0, 255]

    return mask, edges, overlay


# ---------------------------
# Version 2: 套用在 YOLO 預測後流程（假設已輸出 binary mask）
# ---------------------------
def postprocess_yolo_mask(yolo_mask_array):
    edges = cv2.Canny(yolo_mask_array, 50, 150)
    ys, xs = np.where(edges > 0)
    points = np.stack((xs, ys), axis=-1)

    circle_ransac, inliers = fit_circle_ransac(points)
    circle_overlay = np.zeros_like(yolo_mask_array)

    if circle_ransac:
        cx, cy, r = circle_ransac
        theta = np.linspace(0, 2 * np.pi, 360)
        arc_x = (cx + r * np.cos(theta)).astype(np.int32)
        arc_y = (cy + r * np.sin(theta)).astype(np.int32)
        valid = (arc_x >= 0) & (arc_x < yolo_mask_array.shape[1]) & (arc_y >= 0) & (arc_y < yolo_mask_array.shape[0])
        circle_overlay[arc_y[valid], arc_x[valid]] = 255

    return circle_overlay, circle_ransac, inliers

mask, edges, overlay = process_single_image("test_edge.png")

# circle_overlay, circle_ransac, inliers = postprocess_yolo_mask(predicted_mask)
# Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Original Mask")
plt.imshow(mask, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Detected Edge")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Fitted Circle Overlay")
plt.imshow(overlay)
plt.axis("off")
plt.tight_layout()
plt.show()