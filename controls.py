import time
import keyboard as kb

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
        kb.press('left')
        time.sleep(delay)
        kb.release('left')
        time.sleep(delay)

def decrease_angle(steps=1, delay=0.01):
    for _ in range(steps):
        kb.press('right')
        time.sleep(delay)
        kb.release('right')
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
