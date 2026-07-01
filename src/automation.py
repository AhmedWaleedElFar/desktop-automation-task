"""
Automation Phase: Launch Notepad once, fetch blog posts, type/save natively 10 times.
Uses recursive visual search once at initialization for maximum efficiency.
"""

import pyautogui
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from dotenv import load_dotenv

from screenshot import take_screenshot
from screenseeker.recursive_search import RecursiveVisualSearcher
from screenseeker.planner import QwenPlanner, HeuristicPlanner
from screenseeker.grounder import OpenRouterGrounder, SimpleHeuristicGrounder
from screenseeker.verifier import QwenVerifier, SimpleVerifier

load_dotenv()

# Configuration
DESKTOP_FOLDER = Path.home() / "Desktop"
OUTPUT_FOLDER = DESKTOP_FOLDER / "tjm-project"
JSONPLACEHOLDER_API = "https://jsonplaceholder.cypress.io/posts"
NUM_POSTS = 2
CLICK_DELAY = 1.5  # Seconds after clicking icon


class DesktopAutomation:
    """
    Optimized automation pipeline: icon detection (once) -> type -> save natively -> clean slate -> repeat.
    """
    
    def __init__(self, target_app: str = "Notepad", use_vlm: bool = True):
        """Initialize automation search parameters and targets."""
        pyautogui.FAILSAFE = False
        self.target_app = target_app
        # Single timestamp shared across all log files for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if use_vlm:
            planner = QwenPlanner()
            grounder = OpenRouterGrounder()
            verifier = QwenVerifier()
        else:
            planner = HeuristicPlanner()
            grounder = SimpleHeuristicGrounder()
            verifier = SimpleVerifier()
            
        self.searcher = RecursiveVisualSearcher(
            planner=planner,
            grounder=grounder,
            verifier=verifier,
            max_depth=2,
            min_patch_size=100,
            confidence_threshold=0.5,
            run_timestamp=self.run_timestamp
        )
        
        # Ensure clean directory creation
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"✓ Output folder verified: {OUTPUT_FOLDER}")
        print(f"✓ Run timestamp: {self.run_timestamp}")
    
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

    def find_and_click_icon(self) -> Optional[tuple]:
        """Find app icon on desktop using single-pass recursive visual search."""
        print(f"\n🔍 Executing recursive visual search for {self.target_app}...")
        try:
            screenshot = take_screenshot()
            result = self.searcher.search(screenshot, f"{self.target_app}")
            
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
        """
        Type a blog post into the active Notepad window.

        pyautogui.write() silently drops '\\n', so we split the content on
        newlines and press Enter between segments ourselves.

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

            # Split on newlines; type each line with typewrite, press Enter between
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line:
                    pyautogui.typewrite(line, interval=0.01)
                if i < len(lines) - 1:
                    pyautogui.press('enter')
                time.sleep(0.03)

            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"  ✗ Error typing post: {e}")
            return False
    
    def save_post(self, post_id: int) -> bool:
        """
        Save current Notepad content via the Windows Save-As dialog.

        Root cause of 'typing into Notepad instead of dialog':
          pyautogui sends Ctrl+S and immediately moves on.  The OS needs time
          to render the dialog and move focus to it.  We fix this by setting
          pyautogui.PAUSE for the duration of this method — every single
          pyautogui call automatically waits that many seconds after it runs,
          so the dialog always has time to catch up.

        Sequence:
          1. Ctrl+S          → open Save-As dialog   (3 s to be safe)
          2. Ctrl+A + Delete → clear filename field
          3. typewrite       → type  post_{id}.txt
          4. Ctrl+L          → focus address / path bar
          5. typewrite       → type  C:\\Users\\...\\Desktop\\tjm-project
          6. Enter           → navigate dialog to that folder
          7. Enter           → confirm save
        """
        # Save the current global PAUSE and raise it for this method only.
        # This guarantees the OS (and the dialog) keeps up with every keystroke.
        _prev_pause = pyautogui.PAUSE
        pyautogui.PAUSE = 0.4   # 400 ms after every pyautogui call in this block

        try:
            filename  = f"post_{post_id}.txt"
            path_str  = str(OUTPUT_FOLDER)   # e.g. C:\Users\Amany\Desktop\tjm-project

            print(f"  Saving → {filename}  in  {path_str}")

            # 1. Open Save-As dialog and wait generously for it to appear
            pyautogui.hotkey('ctrl', 's')   # PAUSE kicks in after this
            time.sleep(2.5)                 # extra insurance: 2.5 s on top of PAUSE

            # 2. Clear the filename field
            pyautogui.hotkey('ctrl', 'a')   # selects existing text in field
            pyautogui.press('delete')       # removes it

            # 3. Type just the filename
            pyautogui.typewrite(filename, interval=0.05)

            # 4. Focus the address / path bar
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.5)                 # give the bar time to become editable

            # 5. Type the directory path
            pyautogui.typewrite(path_str, interval=0.05)

            # 6. Navigate the dialog to that folder
            pyautogui.press('enter')
            time.sleep(1.5)                 # wait for folder navigation

            # 7. Confirm save
            pyautogui.press('enter')
            time.sleep(1.5)                 # wait for disk write

            print(f"  ✓ Saved {filename}")
            return True

        except Exception as e:
            print(f"  ✗ Save error: {e}")
            pyautogui.press('escape')
            return False

        finally:
            pyautogui.PAUSE = _prev_pause   # always restore, even on exception
    
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
    
    def run_full_automation(self, num_posts: int = NUM_POSTS) -> Dict:
        """Orchestrate the single-window automation loop.

        Flow:
          1. Ground Notepad icon → ABORT if not found
          2. Launch Notepad (once)
          3. For each post:
               a. Paste title + body via clipboard
               b. Ctrl+S  → type filename → Alt+D → paste directory → Enter x2
               c. (if not last post)
                    Ctrl+N   → new blank tab
                    Ctrl+Tab → switch back to the just-saved tab
                    Ctrl+W   → close it  (now the new blank tab has focus)
          4. After last post → Alt+F4 to close Notepad
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

        # ── Fetch posts ───────────────────────────────────────────────────────
        posts = self.fetch_blog_posts(num_posts)
        if not posts:
            print("\u2717 No posts fetched. Aborting.")
            return stats

        # # ── Step 1: Ground icon — abort if not found ──────────────────────────
        print(f"\n🔍 Grounding {self.target_app} icon...")
        screenshot = take_screenshot()
        result = self.searcher.search(screenshot, self.target_app)

        if not result.found:
            print(f"✗ Could not locate {self.target_app} icon. Terminating automation.")
            return stats

        x, y = result.center
        print(f"✓ Found {self.target_app} at ({x}, {y}) with {result.confidence:.0%} confidence")

        # ── Step 2–3: Per-post loop — re-launch Notepad for every post ─────────
        # Ctrl+W closes Notepad after each save, so we must re-click the icon
        # at the start of each iteration to get a fresh window.
        for i, post in enumerate(posts, 1):
            print(f"\n[{i}/{num_posts}] Processing post #{post['id']}: "
                  f"{post['title'][:40]}...")
            try:
                # a. Launch Notepad (fresh window each time)
                pyautogui.hotkey('win', 'd')
                time.sleep(0.5)
                pyautogui.click(x, y, clicks=2, button='left')
                time.sleep(CLICK_DELAY + 1.0)   # wait for Notepad to fully open

                # b. Type post content
                if not self.type_post(post):
                    raise Exception("Failed to type post.")
                time.sleep(0.5)

                # c. Save
                if not self.save_post(post["id"]):
                    raise Exception("Failed to save post.")

                # d. Close Notepad (already saved — Ctrl+W closes cleanly)
                if not self.close_app():
                    raise Exception("Failed to close Notepad.")

                stats["successful_posts"] += 1
                print(f"\u2713 Post #{post['id']} completed.")
                time.sleep(1)

            except Exception as e:
                stats["failed_posts"] += 1
                stats["errors"].append(f"Post {post['id']}: {str(e)}")
                print(f"\u2717 Post #{post['id']} failed: {e}")

                # Best-effort cleanup
                try:
                    pyautogui.press('escape')   # dismiss any open dialog
                    time.sleep(0.3)
                    pyautogui.hotkey('alt', 'f4')
                    time.sleep(0.5)
                    pyautogui.press('n')        # don't save
                    time.sleep(0.5)
                except Exception:
                    pass
                time.sleep(1)

        # ── Summary ──────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("AUTOMATION COMPLETE")
        print("=" * 70)
        print(f"\u2713 Successful: {stats['successful_posts']}/{num_posts}")
        print(f"\u2717 Failed:     {stats['failed_posts']}/{num_posts}")
        print(f"\U0001f4c1 Output:    {OUTPUT_FOLDER}")

        try:
            saved_files = sorted(OUTPUT_FOLDER.glob("post_*.txt"))
            print(f"\U0001f4c4 Files saved ({len(saved_files)}):")
            for f in saved_files:
                print(f"   - {f.name}")
        except Exception:
            pass

        if stats["errors"]:
            print("\n\u26a0\ufe0f  Errors:")
            for err in stats["errors"][:5]:
                print(f"   - {err}")

        return stats


if __name__ == "__main__":
    print("PRODUCTION DESKTOP AUTOMATION RUN")
    print("=" * 70)
    
    automaton = DesktopAutomation(target_app="Notepad", use_vlm=False)
    stats = automaton.run_full_automation(num_posts=10)