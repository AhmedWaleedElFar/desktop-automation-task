"""
Planning Phase: Use Qwen2.5-VL via OpenRouter to predict likely application regions.
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
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class QwenPlanner:
    """Dynamic layout workspace analyzer driven entirely by Qwen2.5-VL."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-vl-72b-instruct")
        if OPENAI_AVAILABLE and self.api_key:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        else:
            self.client = None

    def plan_icon_location(self, screenshot: Image.Image, target_app: str) -> Dict:
        if not self.client:
            raise ValueError("OpenRouter client is uninitialized or API key is missing.")
            
        width, height = screenshot.size
        
        prompt = f"""You are an expert operating system workspace analysis model.
The exact image dimensions provided are {width}x{height} pixels.

Task: Predict 3 to 5 bounding boxes [x1, y1, x2, y2] where the '{target_app}' application icon is likely located on this canvas.

Constraints:
1. Every coordinate bound must map within the image boundaries: X [0 to {width}], Y [0 to {height}].
2. Prioritize standard UI placement zones if the image represents a standard desktop space.
3. Return ONLY a valid JSON object matching the requested schema. No markdown wrapping.

Required JSON Structure:
{{
    "reasoning": "Brief analysis details of icon location signatures inside this {width}x{height} patch.",
    "likely_regions": [
        [x1, y1, x2, y2],
        [x1, y1, x2, y2],
        [x1, y1, x2, y2]
    ],
    "predicted_center": [x, y],
    "confidence": 0.85
}}"""

        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1000
        )
        
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
                
        data = json.loads(text)
        
        # Enforce boundary checking filters
        cleaned_regions = []
        for region in data.get("likely_regions", []):
            rx1 = max(0, min(int(region[0]), width))
            ry1 = max(0, min(int(region[1]), height))
            rx2 = max(rx1 + 5, min(int(region[2]), width))
            ry2 = max(ry1 + 5, min(int(region[3]), height))
            cleaned_regions.append([rx1, ry1, rx2, ry2])
            
        data["likely_regions"] = cleaned_regions
        return data

class HeuristicPlanner:
    """
    Fallback heuristic planner (no API needed).
    Use when Gemini is unavailable or for comparison.
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
            "confidence": 0.5,  # Lower confidence than AI
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
    print("PLANNER TEST - Qwen2.5-VL API")
    print("=" * 70)
    
    # Capture a screenshot
    print("\n[1/2] Capturing desktop screenshot...")
    try:
        img = take_screenshot()
        print(f"      Screenshot size: {img.size}")
    except Exception as e:
        print(f"      Error capturing screenshot: {e}")
        exit(1)
    
    # Test Qwen planner
    print("\n[2/2] Testing Qwen Planner...")
    try:
        planner = QwenPlanner()
        result = planner.plan_icon_location(img, target_app="Notepad")
        
        print("\n✓ Qwen Planner Result:")
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
        print(f"\n✗ Qwen Planner failed: {e}")
        print("  Falling back to Heuristic...")
        
        heuristic = HeuristicPlanner()
        result = heuristic.plan_icon_location(img, target_app="Notepad")
        print(f"\n✓ Heuristic fallback result: {result['predicted_center']}")
    
    print("=" * 70)