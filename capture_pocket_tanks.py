import os
import time
from typing import Optional, Tuple, List, Dict

from PIL import ImageGrab, Image, ImageOps, ImageChops

import win32gui
import win32con
import ctypes
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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


def capture_corners(rect: Tuple[int, int, int, int], outdir: str, size: int = 100) -> List[str]:
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    capture_size = max(1, min(size, width, height))

    boxes = {
        "full": (left, top, right, bottom)
    }

    os.makedirs(outdir, exist_ok=True)
    saved = []
    ts = int(time.time())
    for name, box in boxes.items():
        im = ImageGrab.grab(bbox=box)
        fname = os.path.join(outdir, f"{ts}_{name}.png")
        im.save(fname)
        saved.append(fname)

    return saved

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
        text = pytesseract.image_to_string(cropped_img, config='--psm 7') # Default PSM for general text
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def _normalize_digit_image(img: Image.Image, size=(20, 30)) -> Image.Image:
    img = img.convert("L")
    img.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("L", size, 255)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    bw = canvas.point(lambda p: 0 if p < 128 else 255, "1")
    return bw.convert("L")

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

def image_region_file_to_int(region_path: str, templates_dir: str = "digit_templates\\candidates", max_score=0.25) -> int:
    try:
        img = Image.open(region_path)
        templates = load_digit_templates(templates_dir)
        if not templates and templates_dir.endswith("candidates"):
            templates = load_digit_templates(os.path.dirname(templates_dir))
        if not templates:
            text = pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=0123456789 --psm 7')
            s = ''.join(filter(str.isdigit, text))
            return int(s) if s else -1
        segments = _segment_digits_by_projection(img)
        if not segments:
            text = pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=0123456789 --psm 7')
            s = ''.join(filter(str.isdigit, text))
            return int(s) if s else -1
        digits = []
        for seg in segments:
            d, score = _match_digit_image(seg, templates)
            if d == -1 or score > max_score:
                text = pytesseract.image_to_string(img, config='-c tessedit_char_whitelist=0123456789 --psm 7')
                s = ''.join(filter(str.isdigit, text))
                return int(s) if s else -1
            digits.append(str(d))
        return int(''.join(digits)) if digits else -1
    except Exception:
        return -1

def main() -> None:
    ctypes.windll.user32.SetProcessDPIAware()

    found = find_window_by_title_substring("Pocket Tanks")
    if not found:
        return

    hwnd, actual_title = found

    bring_window_to_front(hwnd)
    time.sleep(0.5)

    rect = get_client_screen_rect(hwnd)
    print(rect)
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    print(left,top,right,bottom)

    size = 100
    if width < size or height < size:
        size = min(width, height)

    saved_image_paths = capture_corners(rect, "pocket_tanks_corners", size=size)

    full_screenshot_path = None
    for path in saved_image_paths:
        if "full.png" in path:
            full_screenshot_path = path
            break

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

        full_img = Image.open(full_screenshot_path)
        ts = int(time.time())
        p1_points_region_path = None
        p2_points_region_path = None

        for name, bbox in all_bboxes.items():
            try:
                cropped_region_img = full_img.crop(bbox)
                region_path = os.path.join("pocket_tanks_corners", f"{ts}_{name}_region.png")
                cropped_region_img.save(region_path)
                print(f"Saved {name} region image to: {region_path}")
                if name == "p1_points":
                    p1_points_region_path = region_path
                elif name == "p2_points":
                    p2_points_region_path = region_path
            except Exception as e:
                print(f"Error saving {name} region image with bbox={bbox}: {e}")

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

        extracted_p1_points = image_region_file_to_int(p1_points_region_path, templates_dir="digit_templates\\candidates") if p1_points_region_path else -1
        print(f"Extracted Player 1 Points: {extracted_p1_points}")

        extracted_p2 = extract_text_from_image(full_screenshot_path, all_bboxes["p2_name"])
        print(f"Extracted Player 2 Name: {extracted_p2}")

        extracted_p2_points = image_region_file_to_int(p2_points_region_path, templates_dir="digit_templates\\candidates") if p2_points_region_path else -1
        print(f"Extracted Player 2 Points: {extracted_p2_points}")

    else:
        print("Full screenshot not found for OCR extraction.")


if __name__ == "__main__":
    main()