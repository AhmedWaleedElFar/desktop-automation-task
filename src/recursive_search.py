"""
Recursive Visual Search: Main VISUALSEARCH algorithm from ScreenSeekeR paper.

Depth-based search with:
- Planning phase (predicts candidate regions)
- Grounding phase (locates within regions using voting)
- Verification phase (confirms if target found)
- Recursion (zooms into best region if uncertain)
"""

from typing import Optional, Tuple, Dict
from PIL import Image
from dataclasses import dataclass

from planner import GeminiPlanner, HeuristicPlanner
from grounder import OpenRouterGrounder, SimpleHeuristicGrounder
from verifier import GeminiVerifier, SimpleVerifier
from scoring import get_best_region


@dataclass
class SearchResult:
    """Result of visual search."""
    found: bool
    center: Tuple[int, int]
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    depth: int
    reasoning: str


class RecursiveVisualSearcher:
    """
    ScreenSeekeR-inspired recursive visual search for UI elements.
    
    Algorithm:
    1. Plan: Predict candidate regions (planning model)
    2. Ground: Locate within regions using voting boxes (grounding model)
    3. Score: Rank regions by Gaussian centrality (scoring algorithm)
    4. Verify: Confirm target matches instruction (verification model)
    5. Recurse: If uncertain, crop best region and retry at deeper level
    6. Terminate: When found OR depth exceeded OR box too small
    """
    
    def __init__(
        self,
        max_depth: int = 2,
        min_patch_size: int = 1280,
        confidence_threshold: float = 0.6,
        use_gemini: bool = True
    ):
        """
        Initialize recursive searcher.
        
        Args:
            max_depth: Maximum recursion depth (default 2)
            min_patch_size: Minimum patch size before terminating (pixels, default 1280)
            confidence_threshold: Confidence needed to accept result (0.0-1.0)
            use_gemini: Use Claude API (True) or heuristics only (False)
        """
        self.max_depth = max_depth
        self.min_patch_size = min_patch_size
        self.confidence_threshold = confidence_threshold
        self.use_gemini = use_gemini
        
        # Initialize components
        if use_gemini:
            self.planner = GeminiPlanner()
            self.grounder = OpenRouterGrounder()
            self.verifier = GeminiVerifier()
        else:
            self.planner = HeuristicPlanner()
            self.grounder = SimpleHeuristicGrounder()
            self.verifier = SimpleVerifier()
        
        # Fallbacks
        self.heuristic_planner = HeuristicPlanner()
        self.heuristic_grounder = SimpleHeuristicGrounder()
        self.heuristic_verifier = SimpleVerifier()
    
    def search(
        self,
        screenshot: Image.Image,
        instruction: str,
        depth: int = 0,
        parent_box: Optional[Tuple[int, int, int, int]] = None
    ) -> SearchResult:
        """
        Recursive visual search for target element.
        
        Args:
            screenshot: Current screenshot (may be cropped at deeper levels)
            instruction: Target description (e.g., "Find Notepad icon")
            depth: Current recursion depth
            parent_box: Parent region coordinates (for coordinate mapping)
        
        Returns:
            SearchResult with found status, coordinates, confidence
        """
        
        # Termination condition 1: Max depth exceeded
        if depth > self.max_depth:
            return SearchResult(
                found=False,
                center=(0, 0),
                confidence=0.0,
                bounding_box=(0, 0, 0, 0),
                depth=depth,
                reasoning="Max recursion depth exceeded"
            )
        
        # Termination condition 2: Patch too small
        if screenshot.size[0] * screenshot.size[1] < self.min_patch_size:
            return SearchResult(
                found=False,
                center=(0, 0),
                confidence=0.0,
                bounding_box=(0, 0, 0, 0),
                depth=depth,
                reasoning=f"Patch size {screenshot.size[0]}x{screenshot.size[1]} below minimum"
            )
        
        print(f"\n{'='*70}")
        print(f"[DEPTH {depth}] Searching for: {instruction}")
        print(f"Screenshot size: {screenshot.size}")
        print(f"{'='*70}")
        
        try:
            # PHASE 1: PLANNING - Predict candidate regions
            print(f"\n[PHASE 1] Planning...")
            try:
                plan = self.planner.plan_icon_location(screenshot, instruction)
            except Exception as e:
                print(f"  Planner failed: {e}, using heuristic...")
                plan = self.heuristic_planner.plan_icon_location(screenshot, instruction)
            
            candidate_regions = plan["likely_regions"]
            planning_confidence = plan["confidence"]
            
            # PHASE 2: GROUNDING - Locate within regions using voting
            print(f"\n[PHASE 2] Grounding...")
            try:
                grounding = self.grounder.ground_icon_in_regions(
                    screenshot, candidate_regions, instruction
                )
            except Exception as e:
                print(f"  Grounder failed: {e}, using heuristic...")
                grounding = self.heuristic_grounder.ground_icon_in_regions(
                    screenshot, candidate_regions, instruction
                )
            
            voting_boxes = grounding["voting_boxes"]
            grounding_confidence = grounding["overall_confidence"]
            
            # PHASE 3: SCORING - Rank regions by voting box centrality
            print(f"\n[PHASE 3] Scoring regions...")
            best_idx, best_score, best_region = get_best_region(
                voting_boxes, candidate_regions, sigma=0.3, use_nms=True
            )
            print(f"  Best region index: {best_idx}")
            print(f"  Best region: {best_region}")
            print(f"  Best score: {best_score:.3f}")
            
            # Get center of best region
            x1, y1, x2, y2 = best_region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Map to parent coordinates if needed
            if parent_box:
                parent_x1, parent_y1, _, _ = parent_box
                center_x += parent_x1
                center_y += parent_y1
                best_region = (
                    x1 + parent_x1,
                    y1 + parent_y1,
                    x2 + parent_x1,
                    y2 + parent_y1
                )
            
            combined_confidence = (planning_confidence + grounding_confidence + best_score) / 3
            
            # PHASE 4: VERIFICATION - Confirm target matches instruction
            print(f"\n[PHASE 4] Verification...")
            cropped = screenshot.crop(best_region)
            try:
                is_target, refined_instr = self.verifier.verify_target(cropped, instruction)
            except Exception as e:
                print(f"  Verifier failed: {e}, using heuristic...")
                is_target, refined_instr = self.heuristic_verifier.verify_target(
                    cropped, instruction
                )
            
            # SUCCESS: Target found with high confidence
            if is_target and combined_confidence >= self.confidence_threshold:
                print(f"\n✅ TARGET FOUND at depth {depth}!")
                print(f"   Center: ({center_x}, {center_y})")
                print(f"   Confidence: {combined_confidence:.0%}")
                
                return SearchResult(
                    found=True,
                    center=(center_x, center_y),
                    confidence=combined_confidence,
                    bounding_box=best_region,
                    depth=depth,
                    reasoning=f"Found and verified at depth {depth}"
                )
            
            # LOW CONFIDENCE: Try deeper recursion
            if not is_target or combined_confidence < self.confidence_threshold:
                print(f"\n⚠️  Confidence too low ({combined_confidence:.0%}), recursing...")
                print(f"   Refined instruction: {refined_instr}")
                
                # Crop best region and recurse
                cropped_screenshot = screenshot.crop(best_region)
                parent_coords = best_region
                
                return self.search(
                    cropped_screenshot,
                    refined_instr,
                    depth=depth + 1,
                    parent_box=parent_coords
                )
        
        except Exception as e:
            print(f"\n❌ Search error at depth {depth}: {e}")
            return SearchResult(
                found=False,
                center=(0, 0),
                confidence=0.0,
                bounding_box=(0, 0, 0, 0),
                depth=depth,
                reasoning=f"Error: {str(e)}"
            )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from screenshot import take_screenshot
    
    print("=" * 70)
    print("RECURSIVE VISUAL SEARCH TEST")
    print("=" * 70)
    
    # Capture a screenshot
    print("\n[1/2] Capturing desktop screenshot...")
    try:
        screenshot = take_screenshot()
        print(f"      Screenshot size: {screenshot.size}")
    except Exception as e:
        print(f"      Error: {e}")
        exit(1)
    
    # Run recursive search
    print("\n[2/2] Running recursive visual search...")
    
    try:
        # Use Claude if available, otherwise fallback to heuristics
        searcher = RecursiveVisualSearcher(
            max_depth=2,
            min_patch_size=1280,
            confidence_threshold=0.5,
            use_gemini=True
        )
        
        result = searcher.search(screenshot, "Find the Notepad application icon")
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULT")
        print(f"{'='*70}")
        print(f"Found: {result.found}")
        print(f"Center: {result.center}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Bounding Box: {result.bounding_box}")
        print(f"Depth: {result.depth}")
        print(f"Reasoning: {result.reasoning}")
        
        # Validate result
        if result.found:
            assert result.center[0] > 0, "X coordinate must be positive"
            assert result.center[1] > 0, "Y coordinate must be positive"
            print(f"\n✅ Recursive search test passed!")
        else:
            print(f"\n⚠️  Target not found (this may be expected if Notepad icon not visible)")
    
    except Exception as e:
        print(f"\n❌ Recursive search failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)