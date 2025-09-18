# wafer_circle_by3dirs.py
# -*- coding: utf-8 -*-
"""
操作說明
- 左鍵：點一下指定一個「方向」（相對於中心的向量）；會沿該方向向外找第一個邊界點
- 右鍵：把「中心」移到目前滑鼠位置
- c：清除目前的三個方向 / 邊界點
- s：存檔 (out/xxx_overlay.png 與 out/xxx_circle.json)
- +/-：調整邊界偵測靈敏度 threshold (亮度差)  -1 / +1
- ,/.：調整步長 step_pixels                              -1 / +1
- n / p：下一張 / 上一張（若設定多張圖片）
- q 或 Esc：離開
"""

import cv2
import numpy as np
import json
from pathlib import Path

# ====== 請放你的影像路徑（可放多張，用 n/p 切換）======
IMAGE_PATHS = [
    r"./1.png",
    r"./1.png",
    r"./1.png",
    # 範例：r"./e1dd2a87-c860-4e94-8af7-c90e9d37fda8.png"
]
# =====================================================

OUT_DIR = Path("out"); OUT_DIR.mkdir(exist_ok=True)

# 互動狀態
state = {
    "center": None,          # 圓心（可右鍵移動；預設為影像中心）
    "dir_clicks": [],        # 使用者點的三個方向參考點（只用來取角度）
    "edge_pts": [],          # 沿方向找到的第一個邊界點
    "threshold": 35,         # 邊界判定的亮度差閾值（越小越敏感）
    "step_pixels": 1,        # 往外掃描時每步長（像素）
    "max_steps": 5000        # 最多走多少步（避免無限）
}

def detect_edge_along_angle(gray, center_xy, angle_deg, threshold=35, step_pixels=1, max_steps=5000):
    """
    從 center_xy 沿 angle_deg 方向往外掃，找到第一個亮度差大於 threshold 的點。
    回傳 (x,y) 或 None。
    """
    h, w = gray.shape
    cx, cy = center_xy
    angle = np.deg2rad(angle_deg)
    dx, dy = np.cos(angle), np.sin(angle)

    # 起始值取中心附近的均值，增加穩定性
    r0 = 3
    y0, x0 = np.ogrid[max(0,cy-r0):min(h,cy+r0+1), max(0,cx-r0):min(w,cx+r0+1)]
    base = int(np.mean(gray[y0, x0]))

    prev = base
    for step in range(1, max_steps+1):
        x = int(round(cx + dx * step * step_pixels))
        y = int(round(cy + dy * step * step_pixels))
        if x < 1 or x >= w-1 or y < 1 or y >= h-1:
            return None  # 走出邊界

        # 使用 3x3 區域平均，避免雜訊
        y1, y2 = y-1, y+2
        x1, x2 = x-1, x+2
        val = int(np.mean(gray[y1:y2, x1:x2]))

        if abs(val - prev) >= threshold:
            return (x, y)
        prev = val
    return None

def circle_from_3pts(p1, p2, p3):
    """三點定圓，回傳 (cx, cy, r)；若共線回傳 None。"""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    temp = x2*x2 + y2*y2
    bc = (x1*x1 + y1*y1 - temp) / 2.0
    cd = (temp - x3*x3 - y3*y3) / 2.0
    det = (x1 - x2)*(y2 - y3) - (x2 - x3)*(y1 - y2)
    if abs(det) < 1e-6:  # 三點幾乎共線
        return None
    cx = (bc*(y2 - y3) - cd*(y1 - y2)) / det
    cy = ((x1 - x2)*cd - (x2 - x3)*bc) / det
    r = np.sqrt((cx - x1)**2 + (cy - y1)**2)
    return (int(round(cx)), int(round(cy)), int(round(r)))

def recompute_edge_for_last_direction(gray, img_shape):
    """對最後一個方向重新掃描邊界。"""
    if not state["dir_clicks"]:
        return
    h, w = img_shape[:2]
    cx, cy = state["center"] or (w//2, h//2)
    px, py = state["dir_clicks"][-1]
    angle = np.degrees(np.arctan2(py - cy, px - cx))
    edge = detect_edge_along_angle(
        gray, (cx, cy), angle,
        threshold=state["threshold"],
        step_pixels=state["step_pixels"],
        max_steps=state["max_steps"]
    )
    if edge is not None:
        state["edge_pts"].append(edge)

def recalc_all_edges(gray, img_shape):
    """依現有三個方向點，重算所有邊界點（例如移動中心後）。"""
    state["edge_pts"].clear()
    for px, py in state["dir_clicks"]:
        h, w = img_shape[:2]
        cx, cy = state["center"] or (w//2, h//2)
        angle = np.degrees(np.arctan2(py - cy, px - cx))
        edge = detect_edge_along_angle(
            gray, (cx, cy), angle,
            threshold=state["threshold"],
            step_pixels=state["step_pixels"],
            max_steps=state["max_steps"]
        )
        if edge is not None:
            state["edge_pts"].append(edge)

def on_mouse(event, x, y, flags, param):
    img, gray = param
    if event == cv2.EVENT_LBUTTONDOWN:
        # 指定「方向」：從中心朝 (x,y) 的角度
        if len(state["dir_clicks"]) < 3:
            state["dir_clicks"].append((x, y))
            recompute_edge_for_last_direction(gray, img.shape)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右鍵：移動中心
        state["center"] = (x, y)
        # 重新計算三個方向的邊界
        recalc_all_edges(gray, img.shape)

def draw_ui(vis):
    h, w = vis.shape[:2]
    cx, cy = state["center"] or (w//2, h//2)

    # 畫中心
    cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)
    cv2.line(vis, (cx-12, cy), (cx+12, cy), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(vis, (cx, cy-12), (cx, cy+12), (0, 255, 255), 1, cv2.LINE_AA)

    # 畫方向射線與邊界點
    for i, (px, py) in enumerate(state["dir_clicks"]):
        angle = np.degrees(np.arctan2(py - cy, px - cx))
        # 畫射線
        length = int(0.6 * max(h, w))
        x2 = int(round(cx + np.cos(np.deg2rad(angle))*length))
        y2 = int(round(cy + np.sin(np.deg2rad(angle))*length))
        cv2.line(vis, (cx, cy), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
        # 畫「方向點」(使用者點的位置，非邊界)
        cv2.circle(vis, (px, py), 4, (160, 160, 160), -1)
    for p in state["edge_pts"]:
        cv2.circle(vis, p, 6, (0, 0, 255), -1)

    # 畫圓（若三個邊界點齊）
    if len(state["edge_pts"]) == 3:
        C = circle_from_3pts(*state["edge_pts"])
        if C:
            cv2.circle(vis, (C[0], C[1]), C[2], (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"cx={C[0]}, cy={C[1]}, r={C[2]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 參數資訊
    info = f"threshold={state['threshold']}  step={state['step_pixels']}  dirs={len(state['dir_clicks'])}/3  edges={len(state['edge_pts'])}/3"
    cv2.putText(vis, info, (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def save_result(img_path, img_vis):
    stem = Path(img_path).stem
    out_img = OUT_DIR / f"{stem}_overlay.png"
    cv2.imwrite(str(out_img), img_vis)
    payload = {
        "image": str(img_path),
        "center": state["center"],
        "dir_clicks": state["dir_clicks"],
        "edge_pts": state["edge_pts"]
    }
    out_json = OUT_DIR / f"{stem}_circle.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_img}")
    print(f"[Saved] {out_json}")

def main():
    paths = [p for p in IMAGE_PATHS if Path(p).exists()]
    if not paths:
        print("請在 IMAGE_PATHS 放上存在的圖檔路徑。")
        return

    idx = 0
    win = "Wafer by 3 Directions  (左鍵:方向  右鍵:移中心  c清除  +/-靈敏度  ,/.步長  s存  n/p換圖  q離開)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        img = cv2.imread(paths[idx])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 初始化中心：若尚未設定，取影像中心
        if state["center"] is None:
            h, w = gray.shape
            state["center"] = (w//2, h//2)

        cv2.setMouseCallback(win, on_mouse, param=(img, gray))

        vis = img.copy()
        draw_ui(vis)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            state["dir_clicks"].clear()
            state["edge_pts"].clear()
        elif key == ord('s'):
            vis2 = img.copy(); draw_ui(vis2); save_result(paths[idx], vis2)
        elif key == ord('n'):
            idx = (idx + 1) % len(paths)
            state["dir_clicks"].clear()
            state["edge_pts"].clear()
            # 保留中心位置以利多張對齊；若要重置中心，取消下一行註解
            # state["center"] = None
        elif key == ord('p'):
            idx = (idx - 1) % len(paths)
            state["dir_clicks"].clear()
            state["edge_pts"].clear()
            # state["center"] = None
        elif key == ord('+') or key == ord('='):
            state["threshold"] = min(255, state["threshold"] + 1)
            # 參數變動後重算
            recalc_all_edges(gray, img.shape)
        elif key == ord('-') or key == ord('_'):
            state["threshold"] = max(1, state["threshold"] - 1)
            recalc_all_edges(gray, img.shape)
        elif key == ord(','):
            state["step_pixels"] = max(1, state["step_pixels"] - 1)
            recalc_all_edges(gray, img.shape)
        elif key == ord('.'):
            state["step_pixels"] = min(10, state["step_pixels"] + 1)
            recalc_all_edges(gray, img.shape)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
