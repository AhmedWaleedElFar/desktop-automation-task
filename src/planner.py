"""
Planning Phase: Use Claude API (vision) to predict likely regions where target app icon is located.
Inspired by ScreenSeekeR's position inference.

Returns structured JSON with candidate regions, reasoning, and confidence.
"""

import json
import os
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠ Warning: anthropic not installed. Install with: uv pip install anthropic")


class ClaudePlanner:
    """
    Planning phase using Claude API (vision) for visual desktop analysis.
    More reliable than Gemini, no quota issues.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude planner.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic not installed. Run: uv pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or parameters.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    
    def plan_icon_location(self, screenshot: Image.Image, target_app: str) -> Dict:
        """
        Use Claude API to predict likely regions where target app icon is located.
        
        Args:
            screenshot: PIL Image of desktop (1920x1080)
            target_app: Name of application to find (e.g., "Notepad", "Word")
        
        Returns:
            Dict with keys:
                - reasoning: LLM's explanation of predictions
                - likely_regions: List of [x1, y1, x2, y2] candidate regions
                - predicted_center: [x, y] of most likely center
                - confidence: Model's confidence (0.0-1.0)
        """
        
        # Convert image to base64
        base64_image = self._image_to_base64(screenshot)
        
        prompt = f"""You are a desktop UI expert analyzing a Windows desktop screenshot (1920x1080 resolution).

Task: Predict where the '{target_app}' application icon is located on this desktop.

Requirements:
1. Desktop icons typically appear in corners, edges, or center areas
2. Analyze the visible desktop layout
3. Consider common Windows conventions (top-left, top-right, bottom areas, center)
4. Return ONLY valid JSON, no markdown or extra text

Respond with ONLY this JSON (no code blocks, no markdown):
{{
    "reasoning": "Brief explanation of why these regions are likely",
    "likely_regions": [
        [x1, y1, x2, y2],
        [x1, y1, x2, y2],
        [x1, y1, x2, y2]
    ],
    "predicted_center": [x, y],
    "confidence": 0.75
}}

Notes:
- Coordinates are in pixels (1920x1080 canvas)
- Each region is [x1, y1, x2, y2] (top-left to bottom-right)
- Provide 3-5 candidate regions ranked by likelihood
- predicted_center is your single best guess
- confidence is 0.0-1.0 (how sure you are)
- Return ONLY the JSON object, nothing else"""
        
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
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Validate output structure
            required_keys = ["reasoning", "likely_regions", "predicted_center", "confidence"]
            if not all(k in result for k in required_keys):
                raise ValueError(f"Missing required keys. Got: {list(result.keys())}")
            
            # Validate and clamp regions
            for i, region in enumerate(result["likely_regions"]):
                if len(region) != 4:
                    raise ValueError(f"Invalid region format: {region}")
                x1, y1, x2, y2 = region
                
                # Clamp to valid bounds
                x1 = max(0, min(x1, 1920))
                x2 = max(x1 + 1, min(x2, 1920))
                y1 = max(0, min(y1, 1080))
                y2 = max(y1 + 1, min(y2, 1080))
                result["likely_regions"][i] = [x1, y1, x2, y2]
            
            print(f"✓ Planning phase complete for '{target_app}' (Claude)")
            print(f"  Reasoning: {result['reasoning'][:80]}...")
            print(f"  Confidence: {result['confidence']:.0%}")
            print(f"  Regions: {len(result['likely_regions'])} candidates")
            print(f"  Best guess: {result['predicted_center']}")
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse Claude response as JSON: {e}")
            print(f"Response: {response_text[:300]}")
            raise
        except Exception as e:
            print(f"✗ Claude API error: {e}")
            raise


class HeuristicPlanner:
    """
    Fallback heuristic planner (no API needed).
    Use when Claude is unavailable or for comparison.
    """
    
    @staticmethod
    def plan_icon_location(screenshot: Image.Image, target_app: str) -> Dict:
        """
        Use heuristic rules to predict icon locations.
        No API call needed.
        
        Args:
            screenshot: PIL Image of desktop
            target_app: Name of application
        
        Returns:
            Dict with predicted regions
        """
        
        width, height = screenshot.size  # Expected: 1920x1080
        
        # Typical desktop icon locations (in order of likelihood)
        regions = [
            (0, 0, 400, 400),                      # Top-left (most common)
            (0, 0, 300, height),                   # Left edge
            (width - 400, 0, width, 400),          # Top-right
            (width // 3, height // 3, 2 * width // 3, 2 * height // 3),  # Center
            (0, height - 400, 400, height),        # Bottom-left
            (width - 400, height - 400, width, height),  # Bottom-right
        ]
        
        result = {
            "reasoning": f"Using heuristic desktop layout knowledge for '{target_app}'",
            "likely_regions": [[x1, y1, x2, y2] for x1, y1, x2, y2 in regions],
            "predicted_center": [width // 2, height // 2],
            "confidence": 0.5,  # Lower confidence than Claude
            "method": "heuristic"
        }
        
        print(f"✓ Heuristic planning complete for '{target_app}'")
        print(f"  Regions: {len(result['likely_regions'])} candidates")
        print(f"  Confidence: {result['confidence']:.0%}")
        
        return result


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from screenshot import take_screenshot
    
    print("=" * 70)
    print("PLANNER TEST - Claude API")
    print("=" * 70)
    
    # Capture a screenshot
    print("\n[1/2] Capturing desktop screenshot...")
    try:
        img = take_screenshot()
        print(f"      Screenshot size: {img.size}")
    except Exception as e:
        print(f"      Error capturing screenshot: {e}")
        exit(1)
    
    # Test Claude planner
    print("\n[2/2] Testing Claude Planner...")
    try:
        planner = ClaudePlanner()
        result = planner.plan_icon_location(img, target_app="Notepad")
        
        print("\n✓ Claude Planner Result:")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Predicted Center: {result['predicted_center']}")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Likely Regions:")
        for i, region in enumerate(result['likely_regions'], 1):
            print(f"    {i}. {region}")
        
        # Validate output
        assert isinstance(result['confidence'], (int, float)), "Confidence must be numeric"
        assert 0 <= result['confidence'] <= 1, "Confidence must be 0-1"
        assert len(result['likely_regions']) > 0, "Must have at least one region"
        
        print("\n✓ Planner test passed!")
    except Exception as e:
        print(f"\n✗ Claude Planner failed: {e}")
        print("  Falling back to Heuristic...")
        
        heuristic = HeuristicPlanner()
        result = heuristic.plan_icon_location(img, target_app="Notepad")
        print(f"\n✓ Heuristic fallback result: {result['predicted_center']}")
    
    print("=" * 70)
