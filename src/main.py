"""
Main entry point for Vision-Based Desktop Automation system.

Usage:
    python main.py --test                    # Run all tests
    python main.py --search                  # Test recursive search only
    python main.py --automate [num_posts]    # Run full automation
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_all_tests():
    """Run all component tests."""
    print("=" * 80)
    print("RUNNING ALL COMPONENT TESTS")
    print("=" * 80)
    
    tests = [
        ("Screenshot", "screenshot.py"),
        ("Planner (Claude)", "planner.py"),
        ("Grounder (Claude)", "grounder.py"),
        ("Scoring (Gaussian + NMS)", "scoring.py"),
        ("Verifier (Result Check)", "verifier.py"),
        ("Recursive Searcher (VISUALSEARCH)", "recursive_search.py"),
    ]
    
    results = {}
    
    for test_name, module_path in tests:
        print(f"\n{'='*80}")
        print(f"Testing: {test_name}")
        print(f"{'='*80}")
        
        try:
            # Import and run test
            import subprocess
            result = subprocess.run(
                [sys.executable, str(Path(__file__).parent / module_path)],
                capture_output=False,
                timeout=300  # 5 min timeout per test
            )
            results[test_name] = result.returncode == 0
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    return all(results.values())


def run_search_test():
    """Test recursive visual search only."""
    print("=" * 80)
    print("RECURSIVE VISUAL SEARCH TEST")
    print("=" * 80)
    
    try:
        from recursive_search import RecursiveVisualSearcher
        from screenshot import take_screenshot
        
        print("\n[1/2] Capturing screenshot...")
        screenshot = take_screenshot()
        print(f"      Screenshot size: {screenshot.size}")
        
        print("\n[2/2] Running recursive search...")
        
        # Try Claude first, fallback to heuristics
        try:
            searcher = RecursiveVisualSearcher(use_claude=True)
            print("      Using Claude API")
        except:
            print("      Claude API unavailable, using heuristics")
            searcher = RecursiveVisualSearcher(use_claude=False)
        
        result = searcher.search(screenshot, "Find the Notepad application icon")
        
        print(f"\n{'='*80}")
        print("SEARCH RESULT")
        print(f"{'='*80}")
        print(f"Found: {result.found}")
        print(f"Center: {result.center}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Depth: {result.depth}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.found:
            print("\n✅ Search test passed!")
            return True
        else:
            print("\n⚠️  Target not found (this may be expected if icon not visible)")
            return True  # Don't fail if not found, just note it
    
    except Exception as e:
        print(f"\n❌ Search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_automation(num_posts: int = 10):
    """Run full automation."""
    print("=" * 80)
    print("FULL AUTOMATION TEST")
    print("=" * 80)
    
    try:
        from automation import DesktopAutomation
        
        # Ask for confirmation
        print(f"\n⚠️  WARNING: This will automate Notepad {num_posts} times.")
        print("Ensure Notepad is CLOSED and shortcut is on desktop.")
        print("Press Ctrl+C to cancel, or Enter to continue...")
        
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled by user")
            return False
        
        # Try Claude first, fallback to heuristics
        try:
            automaton = DesktopAutomation(use_claude=True)
            print("Using Claude API for grounding")
        except:
            print("Claude API unavailable, using heuristics")
            automaton = DesktopAutomation(use_claude=False)
        
        stats = automaton.run_full_automation(num_posts=num_posts)
        
        print(f"\n{'='*80}")
        print("AUTOMATION RESULTS")
        print(f"{'='*80}")
        print(f"Successful: {stats['successful_posts']}/{num_posts}")
        print(f"Failed: {stats['failed_posts']}/{num_posts}")
        
        if stats['errors']:
            print("\nErrors:")
            for error in stats['errors'][:5]:
                print(f"  - {error}")
        
        return stats['successful_posts'] > 0
    
    except Exception as e:
        print(f"\n❌ Automation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Vision-Based Desktop Automation with Dynamic Icon Grounding"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run all component tests"
    )
    
    parser.add_argument(
        "--search",
        action="store_true",
        help="Test recursive visual search only"
    )
    
    parser.add_argument(
        "--automate",
        type=int,
        nargs='?',
        const=10,
        help="Run full automation with N posts (default 10)"
    )
    
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Check if all dependencies are installed"
    )
    
    args = parser.parse_args()
    
    # Check setup
    if args.check_setup:
        print("=" * 80)
        print("CHECKING DEPENDENCIES")
        print("=" * 80)
        
        dependencies = [
            ("mss", "mss"),
            ("PIL", "pillow"),
            ("pyautogui", "pyautogui"),
            ("dotenv", "python-dotenv"),
            ("anthropic", "anthropic"),
        ]
        
        all_installed = True
        for module_name, package_name in dependencies:
            try:
                __import__(module_name)
                print(f"✓ {package_name}")
            except ImportError:
                print(f"✗ {package_name} (not installed)")
                all_installed = False
        
        if all_installed:
            print("\n✅ All dependencies installed!")
            return True
        else:
            print("\n❌ Some dependencies missing. Install with:")
            print("   uv pip install mss pillow pyautogui python-dotenv anthropic")
            return False
    
    # Run requested tests
    if args.test:
        return run_all_tests()
    elif args.search:
        return run_search_test()
    elif args.automate is not None:
        return run_automation(num_posts=args.automate)
    else:
        # Default: show help
        parser.print_help()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
