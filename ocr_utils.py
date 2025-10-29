import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image, ImageChops
from typing import Dict, List, Tuple

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

def _match_digit_image(img: Image.Image, templates: Dict[int, Image.Image]) -> Tuple[int, float]:
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
