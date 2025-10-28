import os
import time
from typing import Optional, Tuple, List, Dict

from PIL import ImageGrab, Image, ImageOps, ImageChops

import win32gui
import win32con
import ctypes
import colorsys
import pytesseract
import cv2
import numpy as np

import keyboard as kb

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def move_left(steps=1, delay=0.01):
    for _ in range(steps):
        kb.press('comma')
        time.sleep(delay)
        kb.release('comma')
        time.sleep(delay)


def move_right(steps=1, delay=0.01):
    for _ in range(steps):
        kb.press('dot')
        time.sleep(delay)
        kb.release('dot')
        time.sleep(delay)


def increase_angle(steps=1, delay=0.01):
    for _ in range(steps):
        kb.press('right')
        time.sleep(delay)
        kb.release('right')
        time.sleep(delay)


def decrease_angle(steps=1, delay=0.01):
    for _ in range(steps):
        kb.press('left')
        time.sleep(delay)
        kb.release('left')
        time.sleep(delay)


def increase_power(steps=1, delay=0.01):
    for _ in range(steps):
        kb.press('up')
        time.sleep(delay)
        kb.release('up')
        time.sleep(delay)


def decrease_power(steps=1, delay=0.01):
    for _ in range(steps):
        kb.press('down')
        time.sleep(delay)
        kb.release('down')
        time.sleep(delay)


def fire(delay=0.01):
    kb.press('space')
    time.sleep(delay)
    kb.release('space')
    time.sleep(delay)
   
# thisss is just experimental stuff havent tested at all
def detect_explosion_coordinates(
    hwnd: int,
    game_bbox: Tuple[int, int, int, int],
    timeout_seconds: float = 5.0,
    threshold: int = 50,
    min_contour_area: int = 100
) -> Optional[Tuple[int, int]]:

    bring_window_to_front(hwnd)
    time.sleep(0.1)

    img_before = ImageGrab.grab(bbox=game_bbox)
    gray_before = cv2.cvtColor(np.array(img_before), cv2.COLOR_RGB2GRAY)

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

def find_window_by_title_substring(sub: str) -> Optional[Tuple[int, str]]:
    sub = (sub or "").lower()
    matches: List[Tuple[int, str]] = []

    def enum_handler(hwnd, ctx):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and sub in title.lower():
            ctx.append((hwnd, title))

    win32gui.EnumWindows(enum_handler, matches)
    if not matches:
        return None
    return matches[0]


def get_window_rect(hwnd: int) -> Tuple[int, int, int, int]:
    return win32gui.GetWindowRect(hwnd)


def get_client_screen_rect(hwnd: int) -> Tuple[int, int, int, int]:
    l, t, r, b = win32gui.GetClientRect(hwnd)
    
    client_tl = win32gui.ClientToScreen(hwnd, (l, t))
    client_br = win32gui.ClientToScreen(hwnd, (r, b))
    
    return (client_tl[0], client_tl[1], client_br[0], client_br[1])

def bring_window_to_front(hwnd: int) -> None:
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    except Exception:
        pass
    try:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    except Exception:
        pass
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        try:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass
    try:
        win32gui.SetActiveWindow(hwnd)
    except Exception:
        pass


def capture_full_screenshot(rect: Tuple[int, int, int, int], outdir: str) -> Optional[str]:
    left, top, right, bottom = rect
    os.makedirs(outdir, exist_ok=True)
    screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
    fname = os.path.join(outdir, "latest_screenshot.png")
    screenshot.save(fname)
    return fname

def extract_digits_from_image(image_path: str, bbox: tuple) -> int:
    try:
        img = Image.open(image_path)
        cropped_img = img.crop(bbox)
        text = pytesseract.image_to_string(cropped_img, config='--psm 7 outputbase digits')
        angle_str = ''.join(filter(str.isdigit, text))

        if angle_str:
            return int(angle_str)
            return -1
    except Exception as e:
        print(f"Error extracting digits: {e}")
        return -1

def extract_text_from_image(image_path: str, bbox: tuple) -> str:
    try:
        img = Image.open(image_path)
        cropped_img = img.crop(bbox)
        text = pytesseract.image_to_string(cropped_img, config='--psm 7') 
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

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

def _normalize_digit_image(img: Image.Image, size=(20, 30)) -> Image.Image:
    img = img.convert("L")
    img.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("L", size, 255)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    bw = canvas.point(lambda p: 0 if p < 128 else 255, "1")
    return bw.convert("L")

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

def highest_green_point_between_tanks(
    image_path: str,
    game_bbox: Tuple[int, int, int, int],
    x1: int,
    x2: int,
    min_surface_y: int = 30
) -> Optional[Tuple[int, int]]:
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

        # Broad green mask; adjust if your terrain hue differs
        lower_green = np.array([35, 40, 40], dtype=np.uint8)   # H in [0,179]
        upper_green = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)

    
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        h, w = mask.shape[:2]

        # clamping x-range to ROI
        x_lo = max(min(x1, x2) - left, 0)
        x_hi = min(max(x1, x2) - left, w - 1)
        if x_hi <= x_lo:
            return None

        best_x = None
        best_y = None

        #this calcualtes first green pixel from top for each column 
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


def find_tank_position(image_path: str, tank_color: Tuple[int, int, int], game_bbox: tuple,
                       hue_tol: float = 0.03, min_sat: float = 0.4, min_val: float = 0.3) -> Optional[Tuple[int, int]]:
    try:
        img = Image.open(image_path)
        game_region = img.crop(game_bbox).convert("RGB")

        width, height = game_region.size

        target_h, target_s, target_v = _rgb_to_hsv_tuple(tank_color)

        pixels = list(game_region.getdata())
        mask = [False] * (width * height)
        for i, (r, g, b) in enumerate(pixels):
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            dh = abs(h - target_h)
            dh = min(dh, 1.0 - dh)
            if dh <= hue_tol and s >= min_sat and v >= min_val:
                mask[i] = True

        if not any(mask):
            return None

        visited = [False] * (width * height)
        largest_cc = None
        largest_size = 0
        largest_bbox = None

        def idx(x, y):
            return y * width + x

        for y in range(height):
            for x in range(width):
                i = idx(x, y)
                if not mask[i] or visited[i]:
                    continue
                # flood fill
                stack = [(x, y)]
                visited[i] = True
                coords = []
                min_x, max_x = x, x
                min_y, max_y = y, y

                while stack:
                    cx, cy = stack.pop()
                    coords.append((cx, cy))
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)

                    # neighbors
                    for nx, ny in ((cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)):
                        if 0 <= nx < width and 0 <= ny < height:
                            ni = idx(nx, ny)
                            if not visited[ni] and mask[ni]:
                                visited[ni] = True
                                stack.append((nx, ny))

                if len(coords) > largest_size:
                    largest_size = len(coords)
                    largest_cc = coords
                    largest_bbox = (min_x, min_y, max_x, max_y)

        if not largest_cc:
            return None

        # Compute centroid of largest component
        sx = sum(p[0] for p in largest_cc)
        sy = sum(p[1] for p in largest_cc)
        cx = int(sx / len(largest_cc))
        cy = int(sy / len(largest_cc))

        screen_x = game_bbox[0] + cx
        screen_y = game_bbox[1] + cy
        return (screen_x, screen_y)
    except Exception:
        return None

def load_digit_templates(template_dir: str) -> Dict[int, Image.Image]:
    templates: Dict[int, Image.Image] = {}
    paths_to_try = [template_dir, os.path.join(template_dir, "candidates")]
    for base in paths_to_try:
        if not os.path.isdir(base):
            continue
        for d in range(10):
            if d in templates:
                continue
            for ext in (".png", ".jpg", ".bmp"):
                p = os.path.join(base, f"{d}{ext}")
                if os.path.isfile(p):
                    try:
                        templates[d] = _normalize_digit_image(Image.open(p))
                    except Exception:
                        pass
                    break
    return templates

def _segment_digits_by_projection(img: Image.Image, min_width=2) -> List[Image.Image]:
    gray = img.convert("L")
    bw = gray.point(lambda p: 0 if p < 128 else 255, "1").convert("L")
    w, h = bw.size
    ink = [1 if bw.crop((x, 0, x+1, h)).getbbox() else 0 for x in range(w)]
    segs = []
    in_seg = False
    s = 0
    for x, v in enumerate(ink):
        if v and not in_seg:
            in_seg = True
            s = x
        elif not v and in_seg:
            if x - s >= min_width:
                segs.append(bw.crop((s, 0, x, h)))
            in_seg = False
    if in_seg:
        if w - s >= min_width:
            segs.append(bw.crop((s, 0, w, h)))
    if not segs and w > 0:
        avg = max(1, w // 3)
        for i in range(0, w, avg):
            segs.append(bw.crop((i, 0, min(i+avg, w), h)))
    return [_normalize_digit_image(s) for s in segs]

def _match_digit_image(img: Image.Image, templates: Dict[int, Image.Image]) -> (int, float):
    best = (-1, float("inf"))
    if not templates:
        return best
    for d, t in templates.items():
        a = img.resize(t.size)
        diff = ImageChops.difference(a, t)
        score = sum(diff.histogram()[1:]) / (t.size[0] * t.size[1])
        if score < best[1]:
            best = (d, score)
    return best

def image_region_to_int(image_path: str, bbox: tuple, templates_dir: str = "digit_templates\\candidates", max_score=0.25) -> int:
    try:
        img = Image.open(image_path)
        region = img.crop(bbox)
        templates = load_digit_templates(templates_dir)
        if not templates and templates_dir.endswith("candidates"):
            parent = os.path.dirname(templates_dir)
            templates = load_digit_templates(parent)
        if not templates:
            return extract_digits_from_image(image_path, bbox)
        segments = _segment_digits_by_projection(region)
        if not segments:
            return extract_digits_from_image(image_path, bbox)
        digits = []
        for seg in segments:
            d, score = _match_digit_image(seg, templates)
            if d == -1 or score > max_score:
                return extract_digits_from_image(image_path, bbox)
            digits.append(str(d))
        return int("".join(digits)) if digits else -1
    except Exception:
        return -1

# this basically annotates the peak , had it for debugging purposes
def annotate_points_on_image(
    image_path: str,
    rect: Tuple[int, int, int, int],
    points: List[Tuple[str, Optional[Tuple[int, int]], Tuple[int, int, int]]],
    out_path: Optional[str] = None
) -> Optional[str]:

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

            size = 12  # larger for visibility
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

def get_state():
    ctypes.windll.user32.SetProcessDPIAware()

    found = find_window_by_title_substring("Pocket Tanks")
    if not found:
        return

    hwnd, actual_title = found

    bring_window_to_front(hwnd)
    time.sleep(0.5)

    rect = get_client_screen_rect(hwnd)
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top

    size = 100
    if width < size or height < size:
        size = min(width, height)

    full_screenshot_path = capture_full_screenshot(rect, "pocket_tanks_corners")

    if full_screenshot_path:
        all_bboxes = {
            "angle": (1000, 975, 1075, 1050),
            "power": (1002, 1127, 1085, 1166),
            "move": (531, 987, 606, 1022),
            "weapon": (550, 1109, 753, 1153),
            "p1_name": (0, 0, 241, 55),
            "p1_points": (0, 70, 205, 116),
            "p2_name": (1215, 4, 1590, 63),
            "p2_points": (1348, 68,1595 , 118),
        }

        game_bbox = (0, 60, 1600, 900)

        full_img = Image.open(full_screenshot_path)
        ts = int(time.time())
        p1_points_region_path = None
        p2_points_region_path = None
        p1_name_region_path = None
        p2_name_region_path = None

        extracted_angle = extract_digits_from_image(full_screenshot_path, all_bboxes["angle"])

        extracted_power = extract_digits_from_image(full_screenshot_path, all_bboxes["power"])

        extracted_move = extract_digits_from_image(full_screenshot_path, all_bboxes["move"])

        extracted_weapon = extract_text_from_image(full_screenshot_path, all_bboxes["weapon"])

        extracted_p1 = extract_text_from_image(full_screenshot_path, all_bboxes["p1_name"])

        extracted_p1_points = image_region_to_int(full_screenshot_path,all_bboxes["p1_points"], templates_dir="digit_templates\\candidates")

        extracted_p2 = extract_text_from_image(full_screenshot_path, all_bboxes["p2_name"])

        extracted_p2_points = image_region_to_int(full_screenshot_path,all_bboxes["p2_points"], templates_dir="digit_templates\\candidates")


        p1_mask, p1_centroid, p1_dom = create_mask_and_centroid_for_player("P1", full_screenshot_path, game_bbox, region_bbox=all_bboxes.get("p1_name"))
        p2_mask, p2_centroid, p2_dom = create_mask_and_centroid_for_player("P2", full_screenshot_path, game_bbox, region_bbox=all_bboxes.get("p2_name"))

        return extracted_angle, extracted_power, extracted_p1_points, extracted_p2_points, p1_centroid, p2_centroid

def main() -> None:
    ctypes.windll.user32.SetProcessDPIAware()

    found = find_window_by_title_substring("Pocket Tanks")
    if not found:
        return

    hwnd, actual_title = found

    bring_window_to_front(hwnd)
    time.sleep(0.5)

    rect = get_client_screen_rect(hwnd)
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top

    size = 100
    if width < size or height < size:
        size = min(width, height)

    full_screenshot_path = capture_full_screenshot(rect, "pocket_tanks_corners")

    if full_screenshot_path:
        all_bboxes = {
            "angle": (1000, 975, 1075, 1050),
            "power": (1002, 1127, 1085, 1166),
            "move": (531, 987, 606, 1022),
            "weapon": (550, 1109, 753, 1153),
            "p1_name": (0, 0, 241, 55),
            "p1_points": (0, 70, 205, 116),
            "p2_name": (1215, 4, 1590, 63),
            "p2_points": (1348, 68,1595 , 118),
        }

        game_bbox = (0, 60, 1600, 900)

        full_img = Image.open(full_screenshot_path)
        ts = int(time.time())
        p1_points_region_path = None
        p2_points_region_path = None
        p1_name_region_path = None
        p2_name_region_path = None

        extracted_angle = extract_digits_from_image(full_screenshot_path, all_bboxes["angle"])
        print(f"Extracted Angle: {extracted_angle}°")

        extracted_power = extract_digits_from_image(full_screenshot_path, all_bboxes["power"])
        print(f"Extracted Power: {extracted_power}")

        extracted_move = extract_digits_from_image(full_screenshot_path, all_bboxes["move"])
        print(f"Extracted Move: {extracted_move}")

        extracted_weapon = extract_text_from_image(full_screenshot_path, all_bboxes["weapon"])
        print(f"Extracted Weapon: {extracted_weapon}")

        extracted_p1 = extract_text_from_image(full_screenshot_path, all_bboxes["p1_name"])
        print(f"Extracted Player 1 Name: {extracted_p1}")

        extracted_p1_points = image_region_to_int(full_screenshot_path,all_bboxes["p1_points"], templates_dir="digit_templates\\candidates")
        print(f"Extracted Player 1 Points: {extracted_p1_points}")

        extracted_p2 = extract_text_from_image(full_screenshot_path, all_bboxes["p2_name"])
        print(f"Extracted Player 2 Name: {extracted_p2}")

        extracted_p2_points = image_region_to_int(full_screenshot_path,all_bboxes["p2_points"], templates_dir="digit_templates\\candidates")
        print(f"Extracted Player 2 Points: {extracted_p2_points}")


        p1_mask, p1_centroid, p1_dom = create_mask_and_centroid_for_player("P1", full_screenshot_path, game_bbox, region_bbox=all_bboxes.get("p1_name"))
        p2_mask, p2_centroid, p2_dom = create_mask_and_centroid_for_player("P2", full_screenshot_path, game_bbox, region_bbox=all_bboxes.get("p2_name"))

        print(f"P1 Position: {p1_centroid}")
        print(f"P2 Position: {p2_centroid}")

        # Find highest green terrain point between tanks
        peak_green = None
        if p1_centroid and p2_centroid:
            peak_green = highest_green_point_between_tanks(
                full_screenshot_path,
                game_bbox,
                p1_centroid[0],
                p2_centroid[0],
                min_surface_y=30
            )
            print(f"Highest green terrain point between tanks: {peak_green}")
        else:
            print("Highest green terrain point: cannot compute (missing tank positions).")

        # Annotate the peak on the image for visual verification
        annot_path = annotate_points_on_image(
            full_screenshot_path,
            rect,
            [("Peak", peak_green, (0, 0, 255))],
            out_path=os.path.join("pocket_tanks_corners", "annotated_latest_screenshot.png")
        )
        if annot_path:
            print(f"Annotated image saved at: {annot_path}")
    

    else:
        print("Full screenshot not found for OCR extraction.")

    decrease_angle(30)
    decrease_power(35)
    fire()


if __name__ == "__main__":
    main()
