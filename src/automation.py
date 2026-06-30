"""
Automation Phase: Launch Notepad once, fetch blog posts, type/save natively 10 times.
Uses recursive visual search once at initialization for maximum efficiency.
"""

import os
import time
import pyautogui
import requests
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from dotenv import load_dotenv

from screenshot import take_screenshot
from recursive_search import RecursiveVisualSearcher

load_dotenv()

# Configuration
DESKTOP_FOLDER = Path.home() / "Desktop"
OUTPUT_FOLDER = DESKTOP_FOLDER / "tjm-project"
JSONPLACEHOLDER_API = "https://jsonplaceholder.cypress.io/posts"
NUM_POSTS = 10
CLICK_DELAY = 1.5  # Seconds after clicking icon


class DesktopAutomation:
    """
    Optimized automation pipeline: icon detection (once) -> type -> save natively -> clean slate -> repeat.
    """
    
    def __init__(self, target_app: str = "Notepad", use_gemini: bool = False):
        """Initialize automation search parameters and targets."""
        self.target_app = target_app
        self.searcher = RecursiveVisualSearcher(
            max_depth=2,
            min_patch_size=100,
            confidence_threshold=0.5
        )
        
        # Ensure clean directory creation
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output folder verified: {OUTPUT_FOLDER}")
    
    def fetch_blog_posts(self, num_posts: int = NUM_POSTS) -> List[Dict]:
        """Fetch blog posts from JSONPlaceholder API."""
        print(f"\n📡 Fetching {num_posts} posts from JSONPlaceholder API...")
        try:
            response = requests.get(JSONPLACEHOLDER_API, timeout=10)
            response.raise_for_status()
            all_posts = response.json()
            posts = all_posts[:num_posts]
            print(f"✓ Cultivated {len(posts)} posts successfully")
            return posts
        except Exception as e:
            print(f"✗ Failed to fetch posts: {e}")
            return []
    
    def _set_clipboard(self, text: str):
        """Helper to safely programmatically set the OS clipboard string frame."""
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()

    def find_and_click_icon(self) -> Optional[tuple]:
        """Find app icon on desktop using single-pass recursive visual search."""
        print(f"\n🔍 Executing recursive visual search for {self.target_app}...")
        try:
            screenshot = take_screenshot()
            result = self.searcher.search(screenshot, f"Find the {self.target_app} application icon")
            
            if not result.found:
                print(f"✗ Could not locate {self.target_app} icon")
                return None
            
            x, y = result.center
            print(f"✓ Found {self.target_app} at ({x}, {y}) with {result.confidence:.0%} confidence")
            
            # Minimize windows once right before clicking to clear overlay conflicts
            pyautogui.hotkey('win', 'd')
            time.sleep(0.5)
            
            # Double-click to launch app
            pyautogui.click(x, y, clicks=2, button='left')
            time.sleep(CLICK_DELAY)
            return (x, y)
        except Exception as e:
            print(f"✗ Error during grounding search loop: {e}")
            return None
    
    def type_post(self, post: Dict) -> bool:
        """Type a blog post into Notepad via safe programmatic clipboard streaming."""
        try:
            title = post.get("title", "Untitled")
            body = post.get("body", "")
            content = f"Title: {title}\n\n{body}"
            
            print(f"  Pasting post payload layout contents: {title[:50]}...")
            
            # Put content into clipboard and paste instantly
            pyautogui.write(content)
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"  ✗ Error streaming post text contents: {e}")
            return False
    
    def save_post(self, post_id: int) -> bool:
        """Save text natively by pasting the exact absolute resolved destination path into Save dialog."""
        try:
            filename = f"post_{post_id}.txt"
            filepath = OUTPUT_FOLDER / filename
            absolute_path = str(filepath.resolve())
            
            print(f"  Saving natively via path registration to: {filename}...")
            
            # Trigger Windows Save file dialogue
            pyautogui.hotkey('ctrl', 's')
            time.sleep(1.5)  # Generous safety delay for native GUI popup rendering
            
            # Copy the absolute path to clipboard and paste it instantly into focused filename line
            self._set_clipboard(absolute_path)
            time.sleep(0.2)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.5)
            
            # Submit save sequence parameters
            pyautogui.press('enter')
            time.sleep(1.2)  # Allow Windows thread safe disk write operations
            return True
        except Exception as e:
            print(f"  ✗ Native dialog save trace error: {e}")
            pyautogui.press('escape')
            return False

    def prepare_next_document(self) -> bool:
        """Instantiates a fresh operational file tab/window sheet avoiding app restarts."""
        try:
            print("  Clearing interface sheet workspace for next post context item...")
            pyautogui.hotkey('ctrl', 'n')
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"  ✗ Interface canvas wipe block initialization error: {e}")
            return False
    
    def close_app(self) -> bool:
        """Gracefully close application container window frame at job completion."""
        try:
            print(f"  Closing {self.target_app} workspace context safely...")
            pyautogui.hotkey('alt', 'f4')
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"  ✗ Application closure execution exception: {e}")
            return False
    
    def run_full_automation(self, num_posts: int = NUM_POSTS) -> Dict:
        """Orchestrate full streamlined visual lifecycle execution pipeline loops."""
        print("\n" + "=" * 70)
        print(f"STARTING STREAMLINED O(1) ENGINE RUN: {self.target_app}")
        print("=" * 70)
        
        stats = {
            "total_posts": num_posts,
            "successful_posts": 0,
            "failed_posts": 0,
            "errors": []
        }
        
        posts = self.fetch_blog_posts(num_posts)
        if not posts:
            print("✗ Core JSON tracking matrix empty. Terminating automated loop.")
            return stats
        
        # Ground and launch target UI container EXACTLY ONCE at execution start
        if not self.find_and_click_icon():
            print("✗ System localization target anchor unverified. Aborting job.")
            return stats
            
        time.sleep(2.5)  # Let application setup dependencies fully mount
        
        for i, post in enumerate(posts, 1):
            print(f"\n[{i}/{num_posts}] Compiling post #{post['id']}: {post['title'][:40]}...")
            try:
                if not self.type_post(post):
                    raise Exception("Failed to push layout payload string context.")
                time.sleep(0.5)
                
                if not self.save_post(post["id"]):
                    raise Exception("Failed Windows File dialogue directory submission.")
                
                # Clear active workspace view (skip on final entry processing step)
                if i < len(posts):
                    if not self.prepare_next_document():
                        raise Exception("Failed wiping active document canvas frame.")
                
                stats["successful_posts"] += 1
                print(f"✓ Post #{post['id']} successfully committed to disk.")
                time.sleep(0.5)
            except Exception as e:
                stats["failed_posts"] += 1
                stats["errors"].append(f"Post {post['id']}: {str(e)}")
                print(f"✗ Post #{post['id']} sequence error: {e}")
                
                # Workspace fallback canvas clear reset trigger
                try:
                    pyautogui.press('escape')
                    time.sleep(0.3)
                    pyautogui.hotkey('ctrl', 'a')
                    pyautogui.press('backspace')
                except:
                    pass
                time.sleep(1)
        
        # Clean shutdown once final payload records are committed
        self.close_app()
        
        print("\n" + "=" * 70)
        print("AUTOMATION PROCESSING STAGE COMPLETE")
        print("=" * 70)
        print(f"✓ Successful: {stats['successful_posts']}/{num_posts}")
        print(f"✗ Failed: {stats['failed_posts']}/{num_posts}")
        return stats


if __name__ == "__main__":
    print("PRODUCTION DESKTOP AUTOMATION RUN")
    print("=" * 70)
    
    automaton = DesktopAutomation(target_app="Notepad", use_gemini=False)
    stats = automaton.run_full_automation(num_posts=NUM_POSTS)