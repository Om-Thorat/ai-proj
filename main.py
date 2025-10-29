import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from state import get_state
from window_utils import find_window_by_title_substring, bring_window_to_front, get_client_screen_rect, capture_full_screenshot
from ocr_utils import extract_digits_from_image, extract_text_from_image, image_region_to_int
from color_utils import create_mask_and_centroid_for_player
from terrain_utils import highest_green_point_between_tanks, annotate_points_on_image, predict_landing
import time
import ctypes
import os
from PIL import Image

def main():
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
            "move": (531, 987, 606, 1022),
            "weapon": (550, 1109, 753, 1153),
            "p1_name": (0, 0, 241, 55),
            "p1_points": (0, 70, 205, 116),
            "p2_name": (1215, 4, 1590, 63),
            "p2_points": (1348, 68,1595 , 118),
        }
        game_bbox = (0, 60, 1600, 900)
        full_img = Image.open(full_screenshot_path)
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
        annot_path = annotate_points_on_image(
            full_screenshot_path,
            rect,
            [("Peak", peak_green, (0, 0, 255))],
            out_path=os.path.join("pocket_tanks_corners", "annotated_latest_screenshot.png")
        )
        if annot_path:
            print(f"Annotated image saved at: {annot_path}")
        print(predict_landing(p1_centroid,extracted_angle,extracted_power))
    else:
        print("Full screenshot not found for OCR extraction.")
if __name__ == "__main__":
    main()
