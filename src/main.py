import pyautogui
import time

pyautogui.hotkey('win', 'd')
time.sleep(1)
# Activate the desktop

pyautogui.click(250, 250, clicks=2, button='left')

pyautogui.hotkey('win', 'd')
time.sleep(1)
# Activate the desktop