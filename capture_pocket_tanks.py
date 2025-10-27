import os
import time
from typing import Optional, Tuple, List

from PIL import ImageGrab, Image

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
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    win32gui.SetForegroundWindow(hwnd)
    win32gui.SetActiveWindow(hwnd)


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
            "p1_points": (0, 70, 210, 107),
            "p2_name": (1215, 4, 1590, 63),
            "p2_points": (1348, 68,1595 , 118),
        }

        full_img = Image.open(full_screenshot_path)
        ts = int(time.time())

        for name, bbox in all_bboxes.items():
            try:
                cropped_region_img = full_img.crop(bbox)
                region_path = os.path.join("pocket_tanks_corners", f"{ts}_{name}_region.png")
                cropped_region_img.save(region_path)
                print(f"Saved {name} region image to: {region_path}")
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

        extracted_p1_points = extract_digits_from_image(full_screenshot_path, all_bboxes["p1_points"])
      
        print(f"Extracted Player 1 Points: {extracted_p1_points}")

        extracted_p2 = extract_text_from_image(full_screenshot_path, all_bboxes["p2_name"])
      
        print(f"Extracted Player 2 Name: {extracted_p2}")

        extracted_p2_points = extract_digits_from_image(full_screenshot_path, all_bboxes["p2_points"])
      
        print(f"Extracted Player 2 Points: {extracted_p2_points}")

    else:
        print("Full screenshot not found for OCR extraction.")


if __name__ == "__main__":
    main()