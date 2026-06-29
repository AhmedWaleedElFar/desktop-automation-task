"""
Planning Phase: Use DeepSeek-V3 to predict likely regions where target app icon is located.
Inspired by ScreenSeekeR's position inference.

Uses Hugging Face Inference API for DeepSeek-V3-0324.
No quota limits, available models, generous free tier.
"""

import json
import os
import time
import base64
import requests
from io import BytesIO
from PIL import Image
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DeepSeekPlanner:
    """
    Planning phase using DeepSeek-V3 via Hugging Face Inference API.
    Highly capable model for visual reasoning about desktop layouts.
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize DeepSeek planner.
        
        Args:
            hf_token: Hugging Face API token. If None, uses HF_TOKEN env var.
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in environment or parameters. "
                           "Get one from https://huggingface.co/settings/tokens")
        
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        self.model = "deepseek-ai/DeepSeek-V3-0324"
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
        Use Qwen2.5-VL to predict likely regions where target app icon is located.
        
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
        
        self._rate_limit()
        
        # Convert image to base64 data URL
        image_base64 = self._image_to_base64(screenshot)
        image_data_url = f"data:image/png;base64,{image_base64}"
        
        # Craft the prompt
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
            # Call Qwen via HF Inference API
            payload = {
                "model": self.model,
                "messages": [
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
                                    "url": image_data_url
                                }
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = response.text
                raise RuntimeError(f"HF API error {response.status_code}: {error_msg}")
            
            result_data = response.json()
            
            if "choices" not in result_data or not result_data["choices"]:
                raise ValueError(f"Invalid API response: {result_data}")
            
            response_text = result_data["choices"][0]["message"]["content"].strip()
            
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
            
            # Validate regions
            for i, region in enumerate(result["likely_regions"]):
                if len(region) != 4:
                    raise ValueError(f"Invalid region format: {region}")
                x1, y1, x2, y2 = region
                
                # Clamp to valid bounds
                if not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
                    x1 = max(0, min(x1, 1920))
                    x2 = max(x1 + 1, min(x2, 1920))
                    y1 = max(0, min(y1, 1080))
                    y2 = max(y1 + 1, min(y2, 1080))
                    result["likely_regions"][i] = [x1, y1, x2, y2]
            
            print(f"✓ Planning phase complete for '{target_app}' (DeepSeek-V3)")
            print(f"  Reasoning: {result['reasoning'][:80]}...")
            print(f"  Confidence: {result['confidence']:.0%}")
            print(f"  Regions: {len(result['likely_regions'])} candidates")
            print(f"  Best guess: {result['predicted_center']}")
            
            return result
        
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse Qwen response as JSON: {e}")
            print(f"Response: {response_text[:300]}")
            raise
        except requests.exceptions.Timeout:
            print("✗ Qwen API request timed out (>60s)")
            raise
        except Exception as e:
            print(f"✗ Qwen API error: {e}")
            raise


class HeuristicPlanner:
    """
    Fallback heuristic planner (no API needed).
    Use when Qwen is unavailable or for comparison.
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
            "confidence": 0.5,  # Lower confidence than Qwen
            "method": "heuristic"
        }
        
        print(f"✓ Heuristic planning complete for '{target_app}'")
        print(f"  Regions: {len(result['likely_regions'])} candidates")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Note: Using fallback (no vision-based analysis)")
        
        return result


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from screenshot import take_screenshot
    
    print("=" * 70)
    print("PLANNING PHASE TEST - DeepSeek-V3 via Hugging Face Inference API")
    print("=" * 70)
    
    # Capture a screenshot
    print("\n[1/2] Capturing desktop screenshot...")
    try:
        img = take_screenshot()
        print(f"      Screenshot size: {img.size}")
    except Exception as e:
        print(f"      Error capturing screenshot: {e}")
        exit(1)
    
    # Test DeepSeek planner
    print("\n[2/2] Testing DeepSeek-V3 Planner...")
    try:
        planner = DeepSeekPlanner()
        result = planner.plan_icon_location(img, target_app="Notepad")
        
        print("\n      DeepSeek Result:")
        print(f"      Reasoning: {result['reasoning']}")
        print(f"      Predicted Center: {result['predicted_center']}")
        print(f"      Confidence: {result['confidence']:.0%}")
        print(f"      Likely Regions:")
        for i, region in enumerate(result['likely_regions'], 1):
            print(f"        {i}. {region}")
        
        print("\n✓ Planning phase test complete!")
    except Exception as e:
        print(f"      ✗ DeepSeek Planner failed: {e}")
        print("\n      Falling back to Heuristic Planner...")
        
        heuristic_planner = HeuristicPlanner()
        heuristic_result = heuristic_planner.plan_icon_location(img, target_app="Notepad")
        
        print("\n      Heuristic Result:")
        print(f"      Reasoning: {heuristic_result['reasoning']}")
        print(f"      Predicted Center: {heuristic_result['predicted_center']}")
        print(f"      Confidence: {heuristic_result['confidence']:.0%}")
        print(f"\n✓ Heuristic planning test complete!")
    
    print("=" * 70)