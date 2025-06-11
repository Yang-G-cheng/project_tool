import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.draw import circle_perimeter
from skimage.transform import hough_circle, hough_circle_peaks
# from sklearn.linear_model import RANSACRegressor

# Step 1: Load sample segmentation mask (mock binary mask for illustration)
# Normally you will load from your model's predicted mask, here we simulate one
mask = np.zeros((512, 512), dtype=np.uint8)
# mask = cv2.imread("test_edge.png", 0)
cv2.circle(mask, (256, 256), 200, 255, 5)  # Simulated circular edge

# Step 2: Extract edges using Canny or direct contours
edges = cv2.Canny(mask, 50, 150)
contours = find_contours(edges, 0.8)

# Step 3: Fit a circle using Hough Transform
hough_radii = np.arange(180, 220, 2)  # Expected radius range
hough_res = hough_circle(edges, hough_radii)
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

# Step 4: Draw fitted circle
circle_overlay = np.zeros_like(mask)
for center_y, center_x, radius in zip(cy, cx, radii):
    rr, cc = circle_perimeter(center_y, center_x, radius)
    valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
    circle_overlay[rr[valid], cc[valid]] = 255

# Step 5: Overlay for visualization
overlay = np.stack([mask, circle_overlay, np.zeros_like(mask)], axis=-1)

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



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# # from sklearn.linear_model import RANSACRegressor

# # Step 1: Create a mock mask (as placeholder for your real prediction mask)
# # mask = np.zeros((512, 512), dtype=np.uint8)
# mask = cv2.imread("test_edge.png", 0)
# cv2.ellipse(mask, (256, 256), (200, 200), 0, 0, 180, 255, 5)  # simulate upper half-circle edge

# # Step 2: Extract edges
# edges = cv2.Canny(mask, 50, 150)
# ys, xs = np.where(edges > 0)
# points = np.stack((xs, ys), axis=-1)

# # Step 3: Fit circle using algebraic method + RANSAC
# def fit_circle_ransac(points, max_trials=1000, residual_threshold=2.0):
#     def calc_circle(p1, p2, p3):
#         A = np.array([
#             [p1[0], p1[1], 1],
#             [p2[0], p2[1], 1],
#             [p3[0], p3[1], 1]
#         ])
#         if np.linalg.matrix_rank(A) < 3:
#             return None
#         A1 = np.linalg.det(np.array([
#             [p1[0]**2 + p1[1]**2, p1[1], 1],
#             [p2[0]**2 + p2[1]**2, p2[1], 1],
#             [p3[0]**2 + p3[1]**2, p3[1], 1]
#         ]))
#         A2 = np.linalg.det(np.array([
#             [p1[0]**2 + p1[1]**2, p1[0], 1],
#             [p2[0]**2 + p2[1]**2, p2[0], 1],
#             [p3[0]**2 + p3[1]**2, p3[0], 1]
#         ]))
#         A3 = np.linalg.det(np.array([
#             [p1[0]**2 + p1[1]**2, p1[0], p1[1]],
#             [p2[0]**2 + p2[1]**2, p2[0], p2[1]],
#             [p3[0]**2 + p3[1]**2, p3[0], p3[1]]
#         ]))
#         A4 = np.linalg.det(A)
#         if A4 == 0:
#             return None
#         cx = 0.5 * A1 / A4
#         cy = -0.5 * A2 / A4
#         r = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
#         return cx, cy, r

#     best_inliers = []
#     best_circle = None
#     for _ in range(max_trials):
#         samples = points[np.random.choice(points.shape[0], 3, replace=False)]
#         circle = calc_circle(*samples)
#         if circle is None:
#             continue
#         cx, cy, r = circle
#         dists = np.abs(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r)
#         inliers = points[dists < residual_threshold]
#         if len(inliers) > len(best_inliers):
#             best_inliers = inliers
#             best_circle = (cx, cy, r)
#     return best_circle, best_inliers

# # Fit using RANSAC circle model
# circle_ransac, inliers = fit_circle_ransac(points)

# # Step 4: Draw fitted arc
# circle_overlay = np.zeros_like(mask)
# if circle_ransac:
#     cx, cy, r = circle_ransac
#     theta = np.linspace(0, np.pi, 180)  # half circle only
#     arc_x = (cx + r * np.cos(theta)).astype(np.int32)
#     arc_y = (cy + r * np.sin(theta)).astype(np.int32)
#     valid = (arc_x >= 0) & (arc_x < mask.shape[1]) & (arc_y >= 0) & (arc_y < mask.shape[0])
#     circle_overlay[arc_y[valid], arc_x[valid]] = 255

# # Step 5: Visualize
# overlay = np.stack([mask, circle_overlay, np.zeros_like(mask)], axis=-1)

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.title("Original Mask")
# plt.imshow(mask, cmap='gray')#, cmap='gray'
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.title("Edge + Inliers")
# edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# for x, y in inliers:
#     cv2.circle(edge_vis, (x, y), 1, (0, 255, 0), -1)
# plt.imshow(edge_vis)
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.title("RANSAC Arc Overlay")
# plt.imshow(overlay)
# plt.axis("off")
# plt.tight_layout()
# plt.show()
