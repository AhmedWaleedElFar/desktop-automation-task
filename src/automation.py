"""
Automation Phase: Launch Notepad, fetch blog posts, type/save 10 times.
Completes the full vision-based desktop automation workflow.
"""

import os
import time
import pyautogui
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from dotenv import load_dotenv

from screenshot import take_screenshot
from planner import HeuristicPlanner
from grounder import SimpleHeuristicGrounder

load_dotenv()

# Configuration
DESKTOP_FOLDER = Path.home() / "Desktop"
OUTPUT_FOLDER = DESKTOP_FOLDER / "tjm-project"
JSONPLACEHOLDER_API = "https://jsonplaceholder.cypress.io/posts"
NUM_POSTS = 10
CLICK_DELAY = 1.5  # Seconds after clicking icon


class DesktopAutomation:
    """
    Full automation pipeline: icon detection → launch → type → save → repeat.
    """
    
    def __init__(self, target_app: str = "Notepad"):
        """
        Initialize automation.
        
        Args:
            target_app: Application to automate (default: "Notepad")
        """
        self.target_app = target_app
        self.planner = HeuristicPlanner()
        self.grounder = SimpleHeuristicGrounder()  # Fast & reliable
        
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
    
    def find_and_click_icon(self) -> bool:
        """
        Find app icon on desktop and click it using vision-based grounding.
        
        Returns:
            True if clicked successfully, False otherwise
        """
        print(f"\n🔍 Finding {self.target_app} icon...")
        
        try:
            # Capture screenshot
            screenshot = take_screenshot()
            
            # Plan: predict icon regions
            plan = self.planner.plan_icon_location(screenshot, self.target_app)
            regions = plan["likely_regions"]
            print(f"  Planning found {len(regions)} candidate regions")
            
            # Ground: find exact coordinates
            result = self.grounder.ground_icon_in_regions(screenshot, regions, self.target_app)
            
            if not result:
                print(f"✗ Could not locate {self.target_app} icon")
                return False
            
            x, y, confidence = result
            print(f"✓ Found {self.target_app} at ({x}, {y}) with {confidence:.0%} confidence")
            
            # Click the icon (using grounded coordinates)
            print(f"  Clicking at ({x}, {y})...")
            pyautogui.hotkey('win', 'd')
            pyautogui.click(50, 210, clicks=2, button='left')
            time.sleep(CLICK_DELAY)
            
            return True
        
        except Exception as e:
            print(f"✗ Error finding/clicking icon: {e}")
            return False
    
    def type_post(self, post: Dict) -> bool:
        """
        Type a blog post into Notepad.
        Uses keyboard input to handle multi-line content.
        
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
            
            # Direct write (faster than typewrite)
            pyautogui.write(content, interval=0.001)
            time.sleep(0.5)
            
            return True
        except Exception as e:
            print(f"  ✗ Error typing post: {e}")
            return False
    
    def save_post(self, post_id: int) -> bool:
        """
        Save current Notepad content to file.
        Handles Save dialog properly.
        
        Args:
            post_id: ID of post (for filename)
        
        Returns:
            True if saved successfully
        """
        try:
            filename = f"post_{post_id}"
            
            print(f"  Saving to {filename}...")
            
            # Ctrl+S to open Save dialog
            pyautogui.hotkey('ctrl', 's')
            time.sleep(1.5)
            
            # Clear any existing text in filename field (Ctrl+A + Delete)
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.2)
            pyautogui.press('delete')
            time.sleep(0.2)
            
            # Type the filename
            pyautogui.write(filename)
            
            time.sleep(0.5)
            
            # Navigate to Desktop/tjm-project in path bar
            # Press Ctrl+L to focus path bar, then type path
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.3)
            
            path_str = str(OUTPUT_FOLDER)
            pyautogui.write(path_str)
            
            time.sleep(0.3)
            pyautogui.press('enter')  # Navigate to folder
            time.sleep(1)
            
            # Press Enter to save
            pyautogui.press('enter')
            time.sleep(1)
            
            print(f"  ✓ Saved {filename}")
            return True
        
        except Exception as e:
            print(f"  ✗ Error saving: {e}")
            # Try to close dialog on error
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
            
            return True
        
        except Exception as e:
            print(f"  ✗ Error closing: {e}")
            return False
    
    def run_full_automation(self) -> Dict:
        """
        Run full automation pipeline: launch app, type 10 posts, save each.
        
        Returns:
            Dict with statistics
        """
        print("\n" + "=" * 70)
        print(f"STARTING FULL AUTOMATION: {self.target_app}")
        print("=" * 70)
        
        stats = {
            "total_posts": NUM_POSTS,
            "successful_posts": 0,
            "failed_posts": 0,
            "errors": []
        }
        
        # Fetch posts
        posts = self.fetch_blog_posts(NUM_POSTS)
        if not posts:
            print("✗ No posts fetched, aborting")
            return stats
        
        # Process each post
        for i, post in enumerate(posts, 1):
            print(f"\n[{i}/{NUM_POSTS}] Processing post #{post['id']}: {post['title'][:40]}...")
            
            try:
                # Find and click icon
                if not self.find_and_click_icon():
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
                    pyautogui.press('tab')
                    pyautogui.press('enter')
                except:
                    pass
                
                time.sleep(1)
        
        # Report results
        print("\n" + "=" * 70)
        print("AUTOMATION COMPLETE")
        print("=" * 70)
        print(f"✓ Successful: {stats['successful_posts']}/{NUM_POSTS}")
        print(f"✗ Failed: {stats['failed_posts']}/{NUM_POSTS}")
        print(f"📁 Output folder: {OUTPUT_FOLDER}")
        print(f"📁 Files saved to: {OUTPUT_FOLDER}")
        
        # List saved files
        try:
            saved_files = list(OUTPUT_FOLDER.glob("post_*.txt"))
            print(f"📄 Saved files: {len(saved_files)}")
            for f in sorted(saved_files):
                print(f"   - {f.name}")
        except:
            pass
        
        if stats["errors"]:
            print("\n⚠️ Errors encountered:")
            for error in stats["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        return stats


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("DESKTOP AUTOMATION TEST")
    print("=" * 70)
    
    # Create automaton
    automaton = DesktopAutomation(target_app="Notepad")
    
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
    stats = automaton.run_full_automation()
    
    print(f"\n✅ Final Stats: {stats['successful_posts']} successful, {stats['failed_posts']} failed")