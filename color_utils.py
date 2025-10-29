from typing import Optional, Tuple
import numpy as np
def create_mask_and_centroid_for_player(player_label: str,
                                        full_screenshot_path: str,
                                        game_bbox: tuple,
                                        region_bbox: Optional[tuple],
                                        hue_tol: float = 0.07,
                                        sat_tol: float = 0.28,
                                        val_tol: float = 0.28) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], Optional[Tuple[int,int,int]]]:
    try:
        dominant = None
        if dominant is None and region_bbox:
            try:
                dominant = get_dominant_color(full_screenshot_path, region_bbox)
            except Exception:
                pass
        if not dominant:
            return None, None, None
        left, top, right, bottom = game_bbox
        sample_bbox = (left, 60, right, 900)
        mask = create_color_mask_for_target_in_image(full_screenshot_path, sample_bbox, dominant,
                                                    hue_tol=hue_tol, sat_tol=sat_tol, val_tol=val_tol)
        if mask is None:
            return None, None, dominant
        centroid = analyze_mask_and_get_centroid(mask, sample_bbox)
        return mask, centroid, dominant
    except Exception:
        return None, None, None
import colorsys
import numpy as np
from PIL import Image
from typing import Optional, Tuple

def get_dominant_color(image_path: str, bbox: tuple) -> Optional[Tuple[int, int, int]]:
    try:
        img = Image.open(image_path)
        cropped_img = img.crop(bbox)
        cropped_img = cropped_img.convert("RGB")
        pixels = cropped_img.getdata()
        filtered_pixels = [p for p in pixels if p != (0, 0, 0) and p != (255, 255, 255)]
        if not filtered_pixels:
            return None
        color_counts = {}
        for pixel in filtered_pixels:
            color_counts[pixel] = color_counts.get(pixel, 0) + 1
        dominant_color = max(color_counts, key=color_counts.get)
        return dominant_color
    except Exception as e:
        print(f"Error getting dominant color: {e}")
        return None

def _rgb_to_hsv_tuple(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return (h, s, v)

def create_color_mask_for_target_in_image(image_path: str, game_bbox: tuple, target_rgb: Tuple[int, int, int],
                                          hue_tol: float = 0.06, sat_tol: float = 0.25, val_tol: float = 0.25) -> Optional[np.ndarray]:
    try:
        img = Image.open(image_path).convert("RGB")
        region = img.crop(game_bbox)
        region_np = np.array(region)
        import cv2
        region_bgr = cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        th, ts, tv = _rgb_to_hsv_tuple(target_rgb)
        target_h = int(th * 179)
        target_s = int(ts * 255)
        target_v = int(tv * 255)
        dh = int(max(1, hue_tol * 179))
        ds = int(max(1, sat_tol * 255))
        dv = int(max(1, val_tol * 255))
        lower_h = (target_h - dh) % 180
        upper_h = (target_h + dh) % 180
        lower_s = max(0, target_s - ds)
        upper_s = min(255, target_s + ds)
        lower_v = max(0, target_v - dv)
        upper_v = min(255, target_v + dv)
        if lower_h <= upper_h:
            lower = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
            upper = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        else:
            lower1 = np.array([0, lower_s, lower_v], dtype=np.uint8)
            upper1 = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
            lower2 = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
            upper2 = np.array([179, upper_s, upper_v], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        return mask
    except Exception as e:
        print(f"Error creating color mask: {e}")
        return None

def analyze_mask_and_get_centroid(mask: np.ndarray, bbox: tuple) -> Optional[Tuple[int, int]]:
    import cv2
    try:
        if mask is None:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area <= 0:
            return None
        M = cv2.moments(largest)
        if M.get('m00', 0) == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        left, top, right, bottom = bbox
        screen_x = left + cx
        screen_y = top + cy
        return (screen_x, screen_y)
    except Exception as e:
        print(f"Error analyzing mask for centroid: {e}")
        return None
