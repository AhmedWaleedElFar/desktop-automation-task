"""
Automation Phase: Launch Notepad, fetch blog posts, type/save 10 times.
Uses recursive visual search (ScreenSeekeR-inspired) for robust icon detection.
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
NUM_POSTS = 1
CLICK_DELAY = 1.5  # Seconds after clicking icon


class DesktopAutomation:
    """
    Full automation pipeline: icon detection (recursive search) → launch → type → save → repeat.
    Uses ScreenSeekeR-inspired visual grounding.
    """
    
    def __init__(self, target_app: str = "Notepad", use_gemini: bool = True):
        """
        Initialize automation.
        
        Args:
            target_app: Application to automate (default: "Notepad")
            use_gemini: Use Claude API (True) or heuristics (False)
        """
        self.target_app = target_app
        self.searcher = RecursiveVisualSearcher(
            max_depth=2,
            min_patch_size=1280,
            confidence_threshold=0.5,
            use_gemini=use_gemini
        )
        
        # Create output folder
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output folder: {OUTPUT_FOLDER}")
    
    def fetch_blog_posts(self, num_posts: int = NUM_POSTS) -> List[Dict]:
        """
        Fetch blog posts from JSONPlaceholder API.
        
        Args:
            num_posts: Number of posts to fetch
        
        Returns:
            List of post dicts with 'id', 'title', 'body'
        """
        print(f"\n📡 Fetching {num_posts} posts from JSONPlaceholder API...")
        try:
            response = requests.get(JSONPLACEHOLDER_API, timeout=10)
            response.raise_for_status()
            all_posts = response.json()
            posts = all_posts[:num_posts]
            print(f"✓ Fetched {len(posts)} posts successfully")
            return posts
        except Exception as e:
            print(f"✗ Failed to fetch posts: {e}")
            return []
    
    def find_and_click_icon(self) -> Optional[tuple]:
        """
        Find app icon on desktop using recursive visual search and click it.
        
        Returns:
            Tuple of (x, y) if clicked successfully, None otherwise
        """
        print(f"\n🔍 Finding {self.target_app} icon using recursive visual search...")
        
        try:
            # Capture screenshot
            screenshot = take_screenshot()
            
            # Run recursive visual search
            result = self.searcher.search(screenshot, f"Find the {self.target_app} application icon")
            
            if not result.found:
                print(f"✗ Could not locate {self.target_app} icon")
                return None
            
            x, y = result.center
            confidence = result.confidence
            
            print(f"✓ Found {self.target_app} at ({x}, {y}) with {confidence:.0%} confidence")
            
            # Click the icon (using grounded coordinates)
            print(f"  Clicking at ({x}, {y})...")
            pyautogui.click(x, y, clicks=2, button='left')
            time.sleep(CLICK_DELAY)
            
            return (x, y)
        
        except Exception as e:
            print(f"✗ Error finding/clicking icon: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def type_post(self, post: Dict) -> bool:
        """
        Type a blog post into Notepad.
        Uses clipboard paste for speed (~0.5s per post).
        
        Args:
            post: Post dict with 'title' and 'body'
        
        Returns:
            True if typed successfully
        """
        try:
            title = post.get("title", "Untitled")
            body = post.get("body", "")
            content = f"Title: {title}\n\n{body}"
            
            print(f"  Typing post: {title[:50]}...")
            
            # Use clipboard paste (much faster than character-by-character)
            import subprocess
            
            # Copy to clipboard (works on Windows)
            try:
                # Use a temporary approach: write to clipboard via Python
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                root.clipboard_clear()
                root.clipboard_append(content)
                root.update()
                root.destroy()
                
                # Paste
                time.sleep(0.1)
                pyautogui.hotkey('ctrl', 'v')
                time.sleep(0.5)
                
            except:
                # Fallback: direct typing (slower)
                pyautogui.write(content, interval=0.001)
                time.sleep(0.5)
            
            return True
        except Exception as e:
            print(f"  ✗ Error typing post: {e}")
            return False
    
    def save_post(self, post_id: int) -> bool:
        """
        Save current Notepad content to file.
        
        Args:
            post_id: ID of post (for filename)
        
        Returns:
            True if saved successfully
        """
        try:
            filename = f"post_{post_id}.txt"
            filepath = OUTPUT_FOLDER / filename
            
            print(f"  Saving to {filename}...")
            
            # Ctrl+A to select all
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.2)
            
            # Ctrl+C to copy
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.2)
            
            # Get clipboard content and save directly
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                content = root.clipboard_get()
                root.destroy()
                
                # Write to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  ✓ Saved {filename}")
                return True
            
            except:
                # Fallback: use file save dialog (slower)
                print(f"  Using Notepad save dialog...")
                pyautogui.hotkey('ctrl', 's')
                time.sleep(1)
                
                pyautogui.write(filename)
                time.sleep(0.3)
                pyautogui.press('enter')
                time.sleep(1)
                
                print(f"  ✓ Saved {filename}")
                return True
        
        except Exception as e:
            print(f"  ✗ Error saving: {e}")
            pyautogui.press('escape')
            return False
    
    def close_app(self) -> bool:
        """
        Close Notepad without saving (we already saved).
        
        Returns:
            True if closed successfully
        """
        try:
            print(f"  Closing {self.target_app}...")
            pyautogui.hotkey('ctrl', 'w')
            time.sleep(0.5)
            
            # If save dialog appears, click "Don't Save"
            try:
                pyautogui.press('tab')
                pyautogui.press('enter')
                time.sleep(0.5)
            except:
                pass
            
            return True
        
        except Exception as e:
            print(f"  ✗ Error closing: {e}")
            return False
    
    def run_full_automation(self, num_posts: int = NUM_POSTS) -> Dict:
        """
        Run full automation pipeline: launch app, type 10 posts, save each.
        
        Args:
            num_posts: Number of posts to process
        
        Returns:
            Dict with statistics
        """
        print("\n" + "=" * 70)
        print(f"STARTING FULL AUTOMATION: {self.target_app}")
        print("=" * 70)
        
        stats = {
            "total_posts": num_posts,
            "successful_posts": 0,
            "failed_posts": 0,
            "errors": []
        }
        
        # Fetch posts
        posts = self.fetch_blog_posts(num_posts)
        if not posts:
            print("✗ No posts fetched, aborting")
            return stats
        
        # Process each post
        for i, post in enumerate(posts, 1):
            print(f"\n[{i}/{num_posts}] Processing post #{post['id']}: {post['title'][:40]}...")
            
            try:
                # Find and click icon
                click_result = self.find_and_click_icon()
                if not click_result:
                    raise Exception("Failed to find/click icon")
                
                # Wait for app to launch
                time.sleep(2)
                
                # Type post
                if not self.type_post(post):
                    raise Exception("Failed to type post")
                
                time.sleep(0.5)
                
                # Save post
                if not self.save_post(post["id"]):
                    raise Exception("Failed to save post")
                
                # Close app
                if not self.close_app():
                    raise Exception("Failed to close app")
                
                stats["successful_posts"] += 1
                print(f"✓ Post #{post['id']} completed successfully")
                
                # Brief pause between iterations
                time.sleep(1)
            
            except Exception as e:
                stats["failed_posts"] += 1
                stats["errors"].append(f"Post {post['id']}: {str(e)}")
                print(f"✗ Post #{post['id']} failed: {e}")
                
                # Try to close on error
                try:
                    pyautogui.hotkey('alt', 'F4')
                    time.sleep(0.5)
                except:
                    pass
                
                time.sleep(1)
        
        # Report results
        print("\n" + "=" * 70)
        print("AUTOMATION COMPLETE")
        print("=" * 70)
        print(f"✓ Successful: {stats['successful_posts']}/{num_posts}")
        print(f"✗ Failed: {stats['failed_posts']}/{num_posts}")
        print(f"📁 Output folder: {OUTPUT_FOLDER}")
        
        # List saved files
        try:
            saved_files = list(OUTPUT_FOLDER.glob("post_*.txt"))
            print(f"📄 Saved files: {len(saved_files)}")
            for f in sorted(saved_files):
                print(f"   - {f.name}")
        except:
            pass
        
        if stats["errors"]:
            print("\n⚠️  Errors encountered:")
            for error in stats["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        return stats


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("DESKTOP AUTOMATION TEST - With Recursive Visual Search")
    print("=" * 70)
    
    # Create automaton (use heuristics for testing, Claude for production)
    automaton = DesktopAutomation(target_app="Notepad", use_gemini=False)
    
    # Run full automation
    print("\n⚠️  WARNING: This will automate Notepad 10 times.")
    print("Ensure Notepad is CLOSED and shortcut is on desktop.")
    print("Press Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled by user")
        exit(0)
    
    # Run automation
    stats = automaton.run_full_automation(num_posts=NUM_POSTS)
    
    print(f"\n✅ Final Stats: {stats['successful_posts']} successful, {stats['failed_posts']} failed")
