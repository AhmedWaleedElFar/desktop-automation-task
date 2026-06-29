import mss
from PIL import Image
import pyautogui
import time

def take_screenshot(output_path=None):
    """
    Capture full desktop screenshot using mss (faster than pyautogui).
    
    Args:
        output_path: Optional path to save screenshot. If None, returns PIL Image.
    
    Returns:
        PIL Image of desktop
    """
    with mss.mss() as sct:

        # Index 1 is primary display
        monitor = sct.monitors[1]

        pyautogui.hotkey('win', 'd')
        time.sleep(1)
        # Activate the desktop

        screenshot = sct.grab(monitor)

        pyautogui.hotkey('win', 'd')
        time.sleep(1)
        # Restore the active window

        image = Image.frombytes('RGB', screenshot.size, screenshot.rgb)

        if output_path:
            image.save(output_path)
            print(f"Screenshot saved to {output_path}")

        return image