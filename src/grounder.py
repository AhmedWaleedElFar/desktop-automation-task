"""
Grounding Phase: Use Claude API (vision) to precisely locate target app icon within predicted regions.
Inspired by ScreenSeekeR's visual grounding stage with voting mechanism.

Returns multiple bounding box predictions (voting boxes) for scoring.
"""

import os
import json
import base64
from io import BytesIO
from typing import List, Optional, Tuple, Dict
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠ Warning: anthropic not installed. Install with: uv pip install anthropic")


class ClaudeGrounder:
    """
    Grounding phase using Claude API (vision) to locate target within regions.
    Returns multiple voting boxes for confidence scoring.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude grounder.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic not installed. Run: uv pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Get one from Anthropic console.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    
    def ground_icon_in_regions(
        self,
        screenshot: Image.Image,
        candidate_regions: List[List[int]],
        target_app: str
    ) -> Optional[Dict]:
        """
        Use Claude API to locate icon within candidate regions.
        Returns voting boxes (multiple predictions) for scoring.
        
        Args:
            screenshot: Full desktop screenshot (PIL Image)
            candidate_regions: List of [x1, y1, x2, y2] regions from planning
            target_app: Name of application (e.g., "Notepad", "Word")
        
        Returns:
            Dict with:
                - center: (x, y) best prediction
                - confidence: 0.0-1.0
                - voting_boxes: List of candidate boxes for scoring
        """
        
        base64_image = self._image_to_base64(screenshot)
        
        # Build region descriptions
        region_descriptions = "\n".join([
            f"Region {i}: [{x1}, {y1}, {x2}, {y2}]"
            for i, (x1, y1, x2, y2) in enumerate(candidate_regions)
        ])
        
        prompt = f"""You are a visual grounding expert. Your task is to locate the exact position of a '{target_app}' icon on this desktop screenshot (1920x1080).

Here are candidate regions where it might be located:
{region_descriptions}

Task: For each candidate region, predict the bounding box of the '{target_app}' icon if it exists there.

Return ONLY this JSON (no markdown or code blocks):
{{
    "reasoning": "Why you think the icon is in these locations",
    "voting_boxes": [
        {{"box": [x1, y1, x2, y2], "confidence": 0.95, "region": 0}},
        {{"box": [x1, y1, x2, y2], "confidence": 0.92, "region": 0}},
        {{"box": [x1, y1, x2, y2], "confidence": 0.85, "region": 1}}
    ],
    "best_center": [x, y],
    "overall_confidence": 0.85
}}

Notes:
- voting_boxes: List of 5-10 predicted bounding boxes (votes)
- Each box is [x1, y1, x2, y2] in pixel coordinates
- confidence: How certain you are about this specific box (0.0-1.0)
- region: Which candidate region this box is from
- best_center: The [x, y] center of your single best prediction
- overall_confidence: Overall confidence in finding the target (0.0-1.0)
- Return ONLY JSON, nothing else"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            # Parse response
            response_text = response.content[0].text.strip()
            
            # Remove markdown
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Validate
            required_keys = ["reasoning", "voting_boxes", "best_center", "overall_confidence"]
            if not all(k in result for k in required_keys):
                raise ValueError(f"Missing required keys. Got: {list(result.keys())}")
            
            # Clamp coordinates
            for box_data in result["voting_boxes"]:
                x1, y1, x2, y2 = box_data["box"]
                x1 = max(0, min(x1, 1920))
                x2 = max(x1 + 1, min(x2, 1920))
                y1 = max(0, min(y1, 1080))
                y2 = max(y1 + 1, min(y2, 1080))
                box_data["box"] = [x1, y1, x2, y2]
            
            print(f"✓ Grounding phase complete for '{target_app}' (Claude)")
            print(f"  Reasoning: {result['reasoning'][:80]}...")
            print(f"  Overall confidence: {result['overall_confidence']:.0%}")
            print(f"  Voting boxes: {len(result['voting_boxes'])} predictions")
            print(f"  Best center: {result['best_center']}")
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse Claude response as JSON: {e}")
            print(f"Response: {response_text[:300]}")
            raise
        except Exception as e:
            print(f"✗ Claude grounder error: {e}")
            raise


class SimpleHeuristicGrounder:
    """
    Ultra-simple fallback: No API calls needed.
    Just return center of best region from planning.
    """
    
    @staticmethod
    def ground_icon_in_regions(
        screenshot: Image.Image,
        candidate_regions: List[List[int]],
        target_app: str
    ) -> Optional[Dict]:
        """Return center of first region."""
        if not candidate_regions:
            return None
        
        print(f"\n📍 Grounding phase (Heuristic): Using best-predicted region for '{target_app}'")
        
        x1, y1, x2, y2 = candidate_regions[0]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        result = {
            "reasoning": "Heuristic fallback: using center of best-predicted region",
            "voting_boxes": [
                {"box": [x1, y1, x2, y2], "confidence": 0.5, "region": 0}
            ],
            "best_center": [center_x, center_y],
            "overall_confidence": 0.5
        }
        
        print(f"  Result: ({center_x}, {center_y}) with 50% confidence")
        
        return result


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from screenshot import take_screenshot
    from planner import ClaudePlanner, HeuristicPlanner
    
    print("=" * 70)
    print("GROUNDER TEST - Claude API with Voting Mechanism")
    print("=" * 70)
    
    # Step 1: Capture screenshot
    print("\n[1/3] Capturing desktop screenshot...")
    try:
        screenshot = take_screenshot()
        print(f"      Screenshot size: {screenshot.size}")
    except Exception as e:
        print(f"      Error: {e}")
        exit(1)
    
    # Step 2: Planning phase
    print("\n[2/3] Planning phase...")
    try:
        planner = ClaudePlanner()
        plan = planner.plan_icon_location(screenshot, target_app="Notepad")
        regions = plan["likely_regions"]
        print(f"      Got {len(regions)} candidate regions")
    except Exception as e:
        print(f"      Claude planner failed: {e}")
        print("      Using heuristic planning...")
        planner = HeuristicPlanner()
        plan = planner.plan_icon_location(screenshot, target_app="Notepad")
        regions = plan["likely_regions"]
    
    # Step 3: Grounding phase
    print("\n[3/3] Grounding phase...")
    try:
        grounder = ClaudeGrounder()
        result = grounder.ground_icon_in_regions(screenshot, regions, target_app="Notepad")
        
        if result:
            print(f"\n✓ Grounding Result:")
            print(f"  Reasoning: {result['reasoning']}")
            print(f"  Best Center: {result['best_center']}")
            print(f"  Overall Confidence: {result['overall_confidence']:.0%}")
            print(f"  Voting Boxes: {len(result['voting_boxes'])}")
            
            # Validate
            assert len(result['voting_boxes']) > 0, "Must have voting boxes"
            assert 0 <= result['overall_confidence'] <= 1, "Confidence must be 0-1"
            assert len(result['best_center']) == 2, "Center must be [x, y]"
            
            print(f"\n✅ Grounder test passed!")
        else:
            print("\n⚠ Grounding returned None")
    
    except Exception as e:
        print(f"\n❌ Grounder error: {e}")
        print("Trying heuristic-only grounder...")
        try:
            grounder = SimpleHeuristicGrounder()
            result = grounder.ground_icon_in_regions(screenshot, regions, target_app="Notepad")
            if result:
                x, y = result['best_center']
                print(f"\n✓ Heuristic grounder: ({x}, {y}) with {result['overall_confidence']:.0%} confidence")
        except Exception as e2:
            print(f"✗ All grounding methods failed: {e2}")
    
    print("\n" + "=" * 70)
