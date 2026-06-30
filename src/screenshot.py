"""
Screenshot Capture: Get clean desktop screenshots for vision processing.
Minimizes all windows to show only desktop background and icons.
"""

import mss
from PIL import Image
import pyautogui
import time
from typing import Optional


def take_screenshot(output_path: Optional[str] = None) -> Image.Image:
    """
    Capture full desktop screenshot using mss (faster than pyautogui).
    Minimizes all windows first to show clean desktop.
    
    Args:
        output_path: Optional path to save screenshot. If None, returns PIL Image only.
    
    Returns:
        PIL Image of desktop (RGB, 1920x1080 or native resolution)
    """
    with mss.mss() as sct:
        # Index 1 is primary display
        monitor = sct.monitors[1]
        
        # Minimize all windows to show desktop
        pyautogui.hotkey('win', 'd')
        time.sleep(1)
        
        # Capture screenshot
        screenshot = sct.grab(monitor)
        
        # Restore windows (press Win+D again to toggle)
        pyautogui.hotkey('win', 'd')
        time.sleep(1)
        
        # Convert to PIL Image
        image = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
        
        # Save if path provided
        if output_path:
            image.save(output_path)
            print(f"Screenshot saved to {output_path}")
        
        return image


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SCREENSHOT CAPTURE TEST")
    print("=" * 70)
    
    print("\n[1/2] Capturing screenshot...")
    try:
        img = take_screenshot()
        print(f"      ✓ Screenshot captured: {img.size}")
        print(f"      Format: {img.format}")
        print(f"      Mode: {img.mode}")
        
        # Verify it's the expected size
        assert img.size[0] > 0, "Width must be > 0"
        assert img.size[1] > 0, "Height must be > 0"
        assert img.mode == 'RGB', "Mode must be RGB"
        
        print("\n[2/2] Saving test screenshot...")
        import tempfile
        import os
        
        # 1. Create a secure temporary path, but close the handle immediately
        fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(fd)  # Release the lock right away
        
        try:
            # 2. Save the image using the path
            img.save(temp_path)
            print(f"      ✓ Saved to {temp_path}")
            
            # Verify file exists
            assert os.path.exists(temp_path), "File must exist"
            file_size = os.path.getsize(temp_path)
            print(f"      File size: {file_size} bytes")
            
        finally:
            # 3. Always clean up the file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        print("\n✅ Screenshot capture test passed!")
    
    except Exception as e:
        print(f"\n❌ Screenshot capture failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
