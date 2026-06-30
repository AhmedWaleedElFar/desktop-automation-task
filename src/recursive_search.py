"""
Recursive Visual Search Coordinator Engine entirely mapped over Qwen.
"""

import os
import traceback
from typing import Optional, Tuple, List, Dict
from PIL import Image, ImageDraw
from dataclasses import dataclass

from planner import QwenPlanner, HeuristicPlanner
from grounder import OpenRouterGrounder, SimpleHeuristicGrounder
from verifier import QwenVerifier, SimpleVerifier
from scoring import get_best_region


@dataclass
class SearchResult:
    found: bool
    center: Tuple[int, int]
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    depth: int
    reasoning: str


class RecursiveVisualSearcher:
    
    def __init__(
        self,
        max_depth: int = 2,
        min_patch_size: int = 100,
        confidence_threshold: float = 0.5,
        log_dir: str = "logs"
    ):
        self.max_depth = max_depth
        self.min_patch_size = min_patch_size
        self.confidence_threshold = confidence_threshold
        
        self.planner_log_dir = os.path.join(log_dir, "planner_regions")
        self.grounder_log_dir = os.path.join(log_dir, "grounder_boxes")
        os.makedirs(self.planner_log_dir, exist_ok=True)
        os.makedirs(self.grounder_log_dir, exist_ok=True)
        
        # Instantiate pure Qwen vision components
        self.planner = QwenPlanner()
        self.grounder = OpenRouterGrounder()
        self.verifier = QwenVerifier()
            
        self.heuristic_planner = HeuristicPlanner()
        self.heuristic_grounder = SimpleHeuristicGrounder()
        self.heuristic_verifier = SimpleVerifier()

    def _save_annotated_image(self, base_img: Image.Image, boxes: List[List[int]], output_path: str):
        annotated = base_img.copy()
        draw = ImageDraw.Draw(annotated)
        for i, box in enumerate(boxes):
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0] + 2, box[1] + 2), f"#{i}", fill="red")
        annotated.save(output_path)

    def search(
        self,
        screenshot: Image.Image,
        instruction: str,
        depth: int = 0,
        parent_box: Optional[Tuple[int, int, int, int]] = None
    ) -> SearchResult:
        
        if depth > self.max_depth:
            return SearchResult(False, (0,0), 0.0, (0,0,0,0), depth, "Max tracking depth recursion exceeded.")
            
        w, h = screenshot.size
        if (w * h) < self.min_patch_size:
            return SearchResult(False, (0,0), 0.0, (0,0,0,0), depth, f"Patch canvas area dropped below operational bounds.")
            
        print(f"\n============== [DEPTH {depth} - QWEN RESOLUTION: {w}x{h}] ==============")
        
        # PHASE 1: PLANNING (QWEN)
        try:
            plan = self.planner.plan_icon_location(screenshot, instruction)
            print(f"✓ Qwen Planner isolated target likelihood neighborhoods.")
        except Exception as e:
            print(f"⚠️ Qwen Planner Exception encountered: {e}")
            traceback.print_exc()
            plan = self.heuristic_planner.plan_icon_location(screenshot, instruction)
            
        candidate_regions = plan["likely_regions"]
        
        planner_img_path = os.path.join(self.planner_log_dir, f"depth_{depth}_regions.png")
        self._save_annotated_image(screenshot, candidate_regions, planner_img_path)
        
        # PHASE 2: GROUNDING (QWEN)
        try:
            grounding = self.grounder.ground_icon_in_regions(screenshot, candidate_regions, instruction)
            print(f"✓ Qwen Grounder extracted precise spatial voting frames.")
        except Exception as e:
            print(f"⚠️ Qwen Grounder Exception encountered: {e}")
            traceback.print_exc()
            grounding = self.heuristic_grounder.ground_icon_in_regions(screenshot, candidate_regions, instruction)
            
        voting_boxes = grounding["voting_boxes"]
        
        grounder_img_path = os.path.join(self.grounder_log_dir, f"depth_{depth}_voting_boxes.png")
        self._save_annotated_image(screenshot, [vb["box"] for vb in voting_boxes], grounder_img_path)
        
        # PHASE 3: SCORING (GAUSSIAN CENTRALITY)
        best_idx, best_score, best_region = get_best_region(voting_boxes, candidate_regions, sigma=0.3, use_nms=True)
        print(f"--> Ranked Region #{best_idx} best with density score: {best_score:.3f}")
        
        rx1, ry1, rx2, ry2 = best_region
        local_cx = (rx1 + rx2) // 2
        local_cy = (ry1 + ry2) // 2
        
        # Extract the focused crop sub-canvas
        cropped_patch = screenshot.crop((rx1, ry1, rx2, ry2))
        
        # PHASE 4: VERIFICATION (QWEN)
        try:
            is_target, refined_instr = self.verifier.verify_target(cropped_patch, instruction)
            print(f"✓ Qwen Verification completed. target_match status: {is_target}")
        except Exception as e:
            print(f"⚠️ Qwen Verifier exception: {e}")
            is_target, refined_instr = self.heuristic_verifier.verify_target(cropped_patch, instruction)
            
        # Reconstruct tracking matrix up to true display space coordinates
        global_x = local_cx
        global_y = local_cy
        global_box = [rx1, ry1, rx2, ry2]
        
        if parent_box:
            global_x += parent_box[0]
            global_y += parent_box[1]
            global_box = [
                rx1 + parent_box[0],
                ry1 + parent_box[1],
                rx2 + parent_box[0],
                ry2 + parent_box[1]
            ]
            
        combined_confidence = (plan["confidence"] + grounding["overall_confidence"]) / 2
        if is_target and combined_confidence >= self.confidence_threshold:
            print(f"🎯 Target confirmed at Location Hook: ({global_x}, {global_y})")
            return SearchResult(True, (global_x, global_y), combined_confidence, tuple(global_box), depth, "Target found and verified by Qwen engine.")
            
        print(f"📉 Low match confidence or verification flagged false. Processing deeper recursion search tier...")
        return self.search(
            screenshot=cropped_patch,
            instruction=refined_instr,
            depth=depth + 1,
            parent_box=tuple(global_box)
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
        # Use gemini if available, otherwise fallback to heuristics
        searcher = RecursiveVisualSearcher(
            max_depth=2,
            min_patch_size=1280,
            confidence_threshold=0.5
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