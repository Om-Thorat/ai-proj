import os
import time
import win32gui
import win32con
import ctypes
from typing import Optional, Tuple, List
from PIL import ImageGrab, Image

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
    screenshot = screenshot.resize((1600, 1200), Image.LANCZOS)
    fname = os.path.join(outdir, "latest_screenshot.png")
    screenshot.save(fname)
    return fname
