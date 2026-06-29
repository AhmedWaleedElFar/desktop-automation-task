"""
Planning Phase: Use Gemini to predict likely regions where target app icon is located.
Inspired by ScreenSeekeR's position inference.
"""

import json
import os
import time
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from screenshot import take_screenshot

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ Warning: google-generativeai not installed. Install with: uv pip install google-generativeai")


class GeminiPlanner:
    """
    Planning phase using Google Gemini to predict icon locations.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize Gemini planner.
        
        Args:
            api_key: Google Gemini API key. If None, uses GEMINI_API_KEY env var.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or parameters")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Prevent rate limiting
    
    def _rate_limit(self):
        """Enforce minimum time between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    
    def plan_icon_location(self, screenshot: Image.Image, target_app: str) -> Dict:
        """
        Use Gemini to predict likely regions where target app icon is located.
        
        Args:
            screenshot: PIL Image of desktop (1920x1080)
            target_app: Name of application to find (e.g., "Notepad", "Word")
        
        Returns:
            Dict with keys:
                - reasoning: LLM's explanation of predictions
                - likely_regions: List of [x1, y1, x2, y2] candidate regions
                - predicted_center: [x, y] of most likely center
                - confidence: Gemini's confidence (0.0-1.0)
        """
        
        self._rate_limit()
        
        # Convert image to base64
        image_base64 = self._image_to_base64(screenshot)
        
        # Craft the prompt
        prompt = f"""You are a desktop UI expert analyzing a Windows desktop screenshot.

Task: Predict where the '{target_app}' application icon is located on this desktop.

Requirements:
1. Desktop icons typically appear in corners, edges, or center areas
2. Analyze the visible desktop layout
3. Consider common Windows conventions (top-left, top-right, bottom areas, center)
4. Return ONLY valid JSON, no markdown or extra text

Respond with ONLY this JSON structure (no code blocks):
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

Note:
- Coordinates are in pixels (screenshot is 1920x1080)
- Each region is [x1, y1, x2, y2] (top-left to bottom-right)
- Provide 3-5 candidate regions ranked by likelihood
- predicted_center is your single best guess
- confidence is 0.0-1.0 (how sure you are)"""
        
        try:
            # Call Gemini API with image
            response = self.model.generate_content(
                [
                    prompt,
                    {
                        "mime_type": "image/png",
                        "data": image_base64,
                    }
                ]
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            result = json.loads(response_text)
            
            # Validate output structure
            required_keys = ["reasoning", "likely_regions", "predicted_center", "confidence"]
            if not all(k in result for k in required_keys):
                raise ValueError(f"Missing required keys. Got: {list(result.keys())}")
            
            # Validate regions
            for region in result["likely_regions"]:
                if len(region) != 4:
                    raise ValueError(f"Invalid region format: {region}")
                x1, y1, x2, y2 = region
                if not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
                    print(f"⚠ Warning: Region out of bounds: {region}, clamping...")
                    x1 = max(0, min(x1, 1920))
                    x2 = max(x1, min(x2, 1920))
                    y1 = max(0, min(y1, 1080))
                    y2 = max(y1, min(y2, 1080))
                    result["likely_regions"][result["likely_regions"].index(region)] = [x1, y1, x2, y2]
            
            print(f"✓ Planning phase complete for {target_app}")
            print(f"  Reasoning: {result['reasoning'][:80]}...")
            print(f"  Confidence: {result['confidence']:.0%}")
            print(f"  Regions: {len(result['likely_regions'])} candidates")
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse Gemini response as JSON: {e}")
            print(f"Response: {response_text[:200]}")
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
        
        # Typical desktop icon locations
        regions = [
            (0, 0, 400, 400),                      # Top-left (most common)
            (0, 0, 300, height),                   # Left edge
            (width - 400, 0, width, 400),          # Top-right
            (width // 3, height // 3, 2 * width // 3, 2 * height // 3),  # Center
            (0, height - 400, 400, height),        # Bottom-left
            (width - 400, height - 400, width, height),  # Bottom-right
        ]
        
        result = {
            "reasoning": f"Using heuristic desktop layout knowledge for {target_app}",
            "likely_regions": [[x1, y1, x2, y2] for x1, y1, x2, y2 in regions],
            "predicted_center": [width // 2, height // 2],
            "confidence": 0.6,  # Lower confidence than Gemini
            "method": "heuristic"
        }
        
        print(f"✓ Heuristic planning complete for {target_app}")
        print(f"  Regions: {len(result['likely_regions'])} candidates")
        print(f"  Confidence: {result['confidence']:.0%}")
        
        return result