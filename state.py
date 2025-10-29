import ctypes
import time
from window_utils import find_window_by_title_substring, bring_window_to_front, get_client_screen_rect
from window_utils import capture_full_screenshot
from ocr_utils import extract_digits_from_image, extract_text_from_image, image_region_to_int
from color_utils import create_mask_and_centroid_for_player

def get_state():
    ctypes.windll.user32.SetProcessDPIAware()
    found = find_window_by_title_substring("Pocket Tanks")
    if not found:
        return
    hwnd, actual_title = found
    bring_window_to_front(hwnd)
    time.sleep(0.5)
    rect = get_client_screen_rect(hwnd)
    full_screenshot_path = capture_full_screenshot(rect, "pocket_tanks_corners")
    if full_screenshot_path:
        all_bboxes = {
            "angle": (1000, 975, 1075, 1050),
            "power": (1002, 1127, 1085, 1166),
            "p1_name": (0, 0, 241, 55),
            "p1_points": (0, 70, 205, 116),
            "p2_name": (1215, 4, 1590, 63),
            "p2_points": (1348, 68,1595 , 118),
        }
        game_bbox = (0, 60, 1600, 900)
        extracted_angle = extract_digits_from_image(full_screenshot_path, all_bboxes["angle"])
        extracted_power = extract_digits_from_image(full_screenshot_path, all_bboxes["power"])
        extracted_p1_points = image_region_to_int(full_screenshot_path,all_bboxes["p1_points"], templates_dir="digit_templates\\candidates")
        extracted_p2_points = image_region_to_int(full_screenshot_path,all_bboxes["p2_points"], templates_dir="digit_templates\\candidates")
        p1_mask, p1_centroid, p1_dom = create_mask_and_centroid_for_player("P1", full_screenshot_path, game_bbox, region_bbox=all_bboxes.get("p1_name"))
        p2_mask, p2_centroid, p2_dom = create_mask_and_centroid_for_player("P2", full_screenshot_path, game_bbox, region_bbox=all_bboxes.get("p2_name"))
        return extracted_angle, extracted_power, extracted_p1_points, extracted_p2_points, p1_centroid, p2_centroid
