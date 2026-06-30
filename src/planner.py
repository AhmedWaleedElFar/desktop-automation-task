"""
Planning Phase: Use Gemini 2.5 Flash (vision) to predict likely regions where target app icon is located.
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
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ Warning: google-genai not installed. Install with: pip install google-genai")


class GeminiPlanner:
    """
    Planning phase using Gemini 2.5 Flash (vision) for visual desktop analysis.
    Leverages Google's permissive free-tier, eliminating quota issues.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini planner.
        
        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY env var.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        
        # The genai.Client will naturally look for the GEMINI_API_KEY env variable,
        # but we pass it explicitly here if custom initialized.
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or parameters.")
        
        self.client = genai.Client(api_key=self.api_key)
    
    def plan_icon_location(self, screenshot: Image.Image, target_app: str) -> Dict:
        """
        Use Gemini 2.5 Flash API to predict likely regions where target app icon is located.
        
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
        
        prompt = f"""You are a desktop UI expert analyzing a Windows desktop screenshot (1920x1080 resolution).

Task: Predict where the '{target_app}' application icon is located on this desktop.

Requirements:
1. Desktop icons typically appear in corners, edges, or center areas
2. Analyze the visible desktop layout
3. Consider common Windows conventions (top-left, top-right, bottom areas, center)
4. Return ONLY valid JSON conforming to the requested schema. Do not add markdown or backticks.

Respond with ONLY this JSON structure:
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
- confidence is 0.0-1.0 (how sure you are)"""
        
        try:
            # We can pass the PIL Image directly inside the content parts list.
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[screenshot, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,  # Lower temperature helps structure coordinates reliably
                )
            )
            
            response_text = response.text.strip()
            
            # Clean up potential markdown wrapping
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
                if response_text.endswith("```"):
                    response_text = response_text[:-3].strip()
            
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
            
            print(f"✓ Planning phase complete for '{target_app}' (Gemini)")
            print(f"  Reasoning: {result['reasoning'][:80]}...")
            print(f"  Confidence: {result['confidence']:.0%}")
            print(f"  Regions: {len(result['likely_regions'])} candidates")
            print(f"  Best guess: {result['predicted_center']}")
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse Gemini response as JSON: {e}")
            print(f"Response: {response_text[:300]}")
            raise
        except Exception as e:
            print(f"✗ Gemini API error: {e}")
            raise


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
    print("PLANNER TEST - Gemini 2.5 Flash API")
    print("=" * 70)
    
    # Capture a screenshot
    print("\n[1/2] Capturing desktop screenshot...")
    try:
        img = take_screenshot()
        print(f"      Screenshot size: {img.size}")
    except Exception as e:
        print(f"      Error capturing screenshot: {e}")
        exit(1)
    
    # Test Gemini planner
    print("\n[2/2] Testing Gemini Planner...")
    try:
        planner = GeminiPlanner()
        result = planner.plan_icon_location(img, target_app="Notepad")
        
        print("\n✓ Gemini Planner Result:")
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
        print(f"\n✗ Gemini Planner failed: {e}")
        print("  Falling back to Heuristic...")
        
        heuristic = HeuristicPlanner()
        result = heuristic.plan_icon_location(img, target_app="Notepad")
        print(f"\n✓ Heuristic fallback result: {result['predicted_center']}")
    
    print("=" * 70)