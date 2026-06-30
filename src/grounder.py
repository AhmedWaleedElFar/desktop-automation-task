"""
Grounding Phase: Use Qwen2.5-VL API via OpenRouter to precisely locate the target app icon within predicted regions.
Inspired by ScreenSeekeR's visual grounding stage with voting mechanisms.

Returns multiple bounding box predictions (voting boxes) for scoring.
"""

import os
import json
import base64
from io import BytesIO
from typing import List, Optional, Dict
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ Warning: openai python client not installed. Install with: pip install openai")


class OpenRouterGrounder:
    """
    Grounding phase using state-of-the-art Qwen2.5-VL models on OpenRouter.
    Provides pixel-precise visual grounding utilizing multiple voting boxes.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenRouter Grounder.
        
        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai SDK is required. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Please set it in your .env file.")
        
        # OpenRouter uses the OpenAI-compatible SDK structure
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        # Best visual grounding model option
        self.model = "qwen/qwen-2.5-vl-72b-instruct"
    
    def _image_to_base64_data_uri(self, image: Image.Image) -> str:
        """Convert PIL Image to a base64 Data URI."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"
    
    def ground_icon_in_regions(
        self,
        screenshot: Image.Image,
        candidate_regions: List[List[int]],
        target_app: str
    ) -> Optional[Dict]:
        """
        Use Qwen2.5-VL to locate target icon inside candidate regions.
        
        Args:
            screenshot: Full desktop screenshot (PIL Image)
            candidate_regions: List of [x1, y1, x2, y2] regions from planning
            target_app: Name of application (e.g., "Notepad", "Word")
        
        Returns:
            Dict with:
                - reasoning: Explanation of visual selection
                - voting_boxes: List of predicted bounding boxes with confidence and region mapping
                - best_center: [x, y] coordinates of the single best prediction
                - overall_confidence: Overall score (0.0-1.0)
        """
        
        # Prepare image data
        image_data_uri = self._image_to_base64_data_uri(screenshot)
        
        # Format region lists
        region_descriptions = "\n".join([
            f"- Region {i}: [{x1}, {y1}, {x2}, {y2}]"
            for i, (x1, y1, x2, y2) in enumerate(candidate_regions)
        ])
        
        # Inside grounder.py (ground_icon_in_regions method)
        width, height = screenshot.size

        prompt = f"""You are a pixel-level visual grounding assistant specializing in GUI automation.
        Your task is to locate the exact bounding box of the '{target_app}' icon within this provided image patch.
        The exact canvas resolution is {width}x{height} pixels.

        We have narrowed down the search area to these candidate regions:
        {region_descriptions}

        Task instructions:
        1. Scan the candidate regions in the provided image.
        2. Predict the exact coordinate bounding boxes [x1, y1, x2, y2] relative to this image space.
        3. Coordinates must strictly reside within pixel bounds (0 to {width} width, 0 to {height} height).

        Respond with ONLY raw, valid JSON conforming to this structure:
        {{
            "reasoning": "Reasoning about icon detections",
            "voting_boxes": [
                {{"box": [x1, y1, x2, y2], "confidence": 0.95, "region": 0}}
            ],
            "best_center": [x, y],
            "overall_confidence": 0.92
        }}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_uri
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean markdown JSON wraps if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Validate output keys
            required_keys = ["reasoning", "voting_boxes", "best_center", "overall_confidence"]
            if not all(k in result for k in required_keys):
                raise ValueError(f"Missing required keys in response schema: {list(result.keys())}")
            
            # Coordinate bounding clamp
            for box_data in result["voting_boxes"]:
                x1, y1, x2, y2 = box_data["box"]
                x1 = max(0, min(x1, 1920))
                x2 = max(x1 + 1, min(x2, 1920))
                y1 = max(0, min(y1, 1080))
                y2 = max(y1 + 1, min(y2, 1080))
                box_data["box"] = [x1, y1, x2, y2]
                
            print(f"✓ Grounding phase complete for '{target_app}' (Qwen2.5-VL)")
            print(f"  Reasoning: {result['reasoning'][:80]}...")
            print(f"  Voting boxes: {len(result['voting_boxes'])} bounding-box records")
            print(f"  Best center click point: {result['best_center']}")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse OpenRouter response: {e}")
            print(f"Raw Output: {response_text[:350]}")
            raise
        except Exception as e:
            print(f"✗ OpenRouter grounder error: {e}")
            raise


class SimpleHeuristicGrounder:
    """Fallback heuristic grounder (no API needed)."""
    @staticmethod
    def ground_icon_in_regions(screenshot: Image.Image, candidate_regions: List[List[int]], target_app: str) -> Optional[Dict]:
        if not candidate_regions:
            return None
        x1, y1, x2, y2 = candidate_regions[0]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return {
            "reasoning": "Heuristic fallback: using center of best-predicted region",
            "voting_boxes": [{"box": [x1, y1, x2, y2], "confidence": 0.5, "region": 0}],
            "best_center": [center_x, center_y],
            "overall_confidence": 0.5
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from screenshot import take_screenshot
    from planner import GeminiPlanner, HeuristicPlanner
    
    print("=" * 70)
    print("GROUNDER TEST - Qwen API with Voting Mechanism")
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
        planner = GeminiPlanner()
        plan = planner.plan_icon_location(screenshot, target_app="Notepad")
        regions = plan["likely_regions"]
        print(f"      Got {len(regions)} candidate regions")
    except Exception as e:
        print(f"      Gemini planner failed: {e}")
        print("      Using heuristic planning...")
        planner = HeuristicPlanner()
        plan = planner.plan_icon_location(screenshot, target_app="Notepad")
        regions = plan["likely_regions"]
    
    # Step 3: Grounding phase
    print("\n[3/3] Grounding phase...")
    try:
        grounder = OpenRouterGrounder()
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
