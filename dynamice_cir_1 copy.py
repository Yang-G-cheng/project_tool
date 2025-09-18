# wafer_circle_tool.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import json
from pathlib import Path

# === 請在此放入你的三張 wafer 影像路徑 ===
IMAGE_PATHS = [
    r"./1.png",
    r"./1.png",
    r"./1.png",
    # 範例：r"./e1dd2a87-c860-4e94-8af7-c90e9d37fda8.png"
]

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

def _auto_circle_by_hough(gray, min_r=0, max_r=0):
    """用 HoughCircles 嘗試找圓。回傳 (x, y, r) 或 None。"""
    # 降噪 + 強化邊緣
    g = cv2.GaussianBlur(gray, (9, 9), 2)
    # dp: 累積器解析度反比; param1: Canny 高閾值; param2: 累積器閾值(越小越容易出現假陽)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray.shape[0] // 4,
        param1=120, param2=60, minRadius=min_r, maxRadius=max_r or 0
    )
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype(int)
    # 取半徑最大者（wafer 外圈通常最大）
    x, y, r = max(circles, key=lambda c: c[2])
    return int(x), int(y), int(r)

def _auto_circle_by_contour(gray):
    """以輪廓 + 最小外接圓估計。回傳 (x, y, r)。"""
    # 自適應二值化強化 wafer 與背景的差異
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 50, 150)
    # 膨脹讓外圈邊緣更連續
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # 取面積最大輪廓
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    return int(round(x)), int(round(y)), int(round(r))

def auto_detect_circle(img):
    """綜合法：先 Hough，再 contour。皆失敗則回傳影像中心 + 估計半徑。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 先嘗試 Hough
    circle = _auto_circle_by_hough(gray)
    if circle is None:
        # 再試輪廓法
        circle = _auto_circle_by_contour(gray)

    if circle is None:
        # 兩法都失敗 → 用影像中心 + 估半徑
        r0 = int(0.48 * min(h, w) / 2 * 2)  # 粗估
        circle = (w // 2, h // 2, r0)

    return circle  # (cx, cy, r)

def draw_overlay(base_img, circle, color=(0, 255, 0)):
    """在影像上畫出外圈、中心十字、文字。回傳可視化影像。"""
    cx, cy, r = circle
    vis = base_img.copy()
    # 外圈
    cv2.circle(vis, (cx, cy), max(r, 1), color, 2, lineType=cv2.LINE_AA)
    # 中心十字
    cv2.line(vis, (cx - 15, cy), (cx + 15, cy), color, 1, cv2.LINE_AA)
    cv2.line(vis, (cx, cy - 15), (cx, cy + 15), color, 1, cv2.LINE_AA)
    # 資訊
    text = f"cx={cx}, cy={cy}, r={r}"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return vis

def save_result(img_path, img_vis, circle):
    """儲存覆蓋後影像與參數 JSON。"""
    stem = Path(img_path).stem
    out_img = OUT_DIR / f"{stem}_overlay.png"
    out_json = OUT_DIR / f"{stem}_circle.json"
    cv2.imwrite(str(out_img), img_vis)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"image": str(img_path), "cx": int(circle[0]), "cy": int(circle[1]), "r": int(circle[2])},
            f, ensure_ascii=False, indent=2
        )
    print(f"[Saved] {out_img}")
    print(f"[Saved] {out_json}")

def main():
    paths = [p for p in IMAGE_PATHS if Path(p).exists()]
    if not paths:
        print("請先在 IMAGE_PATHS 填入存在的影像路徑。")
        return

    idx = 0
    window = "Wafer Circle (arrow:move  []:radius  a:auto  r:reset  s:save  n/p:next/prev  q:quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # 載入第一張
    img = cv2.imread(paths[idx])
    base_circle = auto_detect_circle(img)
    cur_circle = list(base_circle)

    while True:
        vis = draw_overlay(img, tuple(cur_circle))
        cv2.imshow(window, vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break

        # 微調中心
        elif key == 81 or key == ord('h'):  # ← 或 h
            cur_circle[0] -= 1
        elif key == 83 or key == ord('l'):  # → 或 l
            cur_circle[0] += 1
        elif key == 82 or key == ord('k'):  # ↑ 或 k
            cur_circle[1] -= 1
        elif key == 84 or key == ord('j'):  # ↓ 或 j
            cur_circle[1] += 1

        # 半徑
        elif key == ord('['):
            cur_circle[2] = max(1, cur_circle[2] - 1)
        elif key == ord(']'):
            cur_circle[2] += 1

        # 重新自動偵測
        elif key == ord('a'):
            base_circle = auto_detect_circle(img)
            cur_circle = list(base_circle)

        # 重置到自動偵測結果
        elif key == ord('r'):
            cur_circle = list(base_circle)

        # 存檔
        elif key == ord('s'):
            save_result(paths[idx], draw_overlay(img, tuple(cur_circle)), tuple(cur_circle))

        # 下一張 / 上一張
        elif key == ord('n') or key == ord('p'):
            if key == ord('n'):
                idx = (idx + 1) % len(paths)
            else:
                idx = (idx - 1) % len(paths)
            img = cv2.imread(paths[idx])
            base_circle = auto_detect_circle(img)
            cur_circle = list(base_circle)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
