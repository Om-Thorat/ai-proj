import numpy as np
from typing import Optional, Tuple, List
import math

def highest_green_point_between_tanks(image_path: str, game_bbox: Tuple[int, int, int, int], x1: int, x2: int, min_surface_y: int = 30) -> Optional[Tuple[int, int]]:
    import cv2
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        left, top, right, bottom = game_bbox
        left = max(0, left); top = max(0, top)
        right = min(img.shape[1], right); bottom = min(img.shape[0], bottom)
        if right <= left or bottom <= top:
            return None
        roi = img[top:bottom, left:right]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        h, w = mask.shape[:2]
        x_lo = max(min(x1, x2) - left, 0)
        x_hi = min(max(x1, x2) - left, w - 1)
        if x_hi <= x_lo:
            return None
        best_x = None
        best_y = None
        for x in range(x_lo, x_hi + 1):
            col = mask[:, x]
            ys = np.where(col > 0)[0]
            if ys.size == 0:
                continue
            y = int(ys[0])
            if y < min_surface_y:
                continue
            if best_y is None or y < best_y:
                best_y = y
                best_x = x
        if best_x is None:
            return None
        return (left + best_x, top + best_y)
    except Exception as e:
        print(f"Error finding highest green point: {e}")
        return None

def annotate_points_on_image(image_path: str, rect: Tuple[int, int, int, int], points: List[Tuple[str, Optional[Tuple[int, int]], Tuple[int, int, int]]], out_path: Optional[str] = None) -> Optional[str]:
    import cv2
    import os
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        for label, pt, color in points:
            if not pt:
                continue
            x = int(pt[0])
            y = int(pt[1])
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            size = 12
            thickness = 2
            cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)
            cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if out_path is None:
            out_dir = os.path.dirname(image_path)
            out_path = os.path.join(out_dir, "annotated_latest_screenshot.png")
        cv2.imwrite(out_path, img)
        return out_path
    except Exception as e:
        print(f"Error annotating image: {e}")
        return None

def detect_explosion_coordinates(hwnd: int, game_bbox: Tuple[int, int, int, int], timeout_seconds: float = 5.0, threshold: int = 50, min_contour_area: int = 100) -> Optional[Tuple[int, int]]:
    import cv2
    import time
    from window_utils import bring_window_to_front
    from PIL import ImageGrab
    bring_window_to_front(hwnd)
    time.sleep(0.1)
    img_before = ImageGrab.grab(bbox=game_bbox)
    gray_before = cv2.cvtColor(np.array(img_before), cv2.COLOR_RGB2GRAY)
    from controls import fire
    fire()
    print("--- Fired! Watching for impact... ---")
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        img_after = ImageGrab.grab(bbox=game_bbox)
        gray_after = cv2.cvtColor(np.array(img_after), cv2.COLOR_RGB2GRAY)
        frame_delta = cv2.absdiff(gray_before, gray_after)
        _, thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + game_bbox[0]
                    cy = int(M["m01"] / M["m00"]) + game_bbox[1]
                    return (cx, cy)
        time.sleep(0.05)
    return None

def predict_landing(p1_pos, angle_deg, power, gravity=9.81):
    c = 0.0140143 * (1.297212032 ** (10.0 - 0.1 * power))
    x0_s, y0_s = p1_pos
    x0 = x0_s * c
    y0 = y0_s * c
    theta = math.radians(angle_deg)
    k = 0.6862496565988354
    v = k / c
    vx = v * math.cos(theta)
    vy = v * math.sin(theta)
    t = (2.0 * vy) / gravity
    x_land = x0 + vx * t
    y_land = y0
    return (x_land / c, y_land / c)
