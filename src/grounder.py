"""
Grounding Phase: Use HF-hosted OpenCLIP to precisely locate target app icon within predicted regions.
No local PyTorch needed - runs entirely via HF Inference API.
Inspired by ScreenSeekeR's visual grounding stage.
"""

import os
import base64
import requests
from io import BytesIO
from typing import List, Optional, Tuple
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class HFCLIPGrounder:
    """
    Grounding phase using OpenCLIP via Hugging Face Inference API.
    No local dependencies - pure API-based zero-shot image classification.
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize HF OpenCLIP grounder.
        
        Args:
            hf_token: Hugging Face API token. If None, uses HF_TOKEN env var.
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found. Get one from https://huggingface.co/settings/tokens")
        
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        # Using a premier OpenCLIP model hosted on HF
        self.clip_api_url = "https://api-inference.huggingface.co/models/laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    
    def _image_to_base64(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def ground_icon_in_regions(
        self,
        screenshot: Image.Image,
        candidate_regions: List[List[int]],
        target_app: str
    ) -> Optional[Tuple[int, int, float]]:
        """
        Use HF-hosted OpenCLIP to locate icon within candidate regions.
        
        Args:
            screenshot: Full desktop screenshot (PIL Image)
            candidate_regions: List of [x1, y1, x2, y2] regions from planning
            target_app: Name of application (e.g., "Notepad", "Word")
        
        Returns:
            Tuple of (center_x, center_y, confidence) or None if not found
        """
        
        best_match = None
        best_score = 0.0
        
        print(f"\n🔍 Grounding phase (HF OpenCLIP): Searching {len(candidate_regions)} regions for '{target_app}'")
        
        # Text labels for OpenCLIP to evaluate against the crop
        text_descriptions = [
            f"a photo of a {target_app} application icon",
            f"a photo of a {target_app} app shortcut",
            f"a photo of a {target_app} desktop icon",
            "a random part of a computer desktop wallpaper or UI",  # Negative constraint label to balance softmax
        ]
        
        # Test each candidate region
        for i, region in enumerate(candidate_regions, 1):
            x1, y1, x2, y2 = region
            cropped = screenshot.crop((x1, y1, x2, y2))
            crop_width = x2 - x1
            crop_height = y2 - y1
            
            print(f"  [{i}/{len(candidate_regions)}] Region {region}...", end=" ", flush=True)
            
            try:
                # Convert image to bytes
                image_bytes = self._image_to_base64(cropped)
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                
                # Query HF OpenCLIP API using the standard Zero-Shot Classification payload
                response = requests.post(
                    self.clip_api_url,
                    headers=self.headers,
                    json={
                        "image": base64_image,
                        "parameters": {"candidate_labels": text_descriptions}
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"HF API error {response.status_code}: {response.text[:200]}")
                
                result = response.json()
                
                # HF zero-shot-image-classification returns a list of items: [{'score': float, 'label': str}, ...]
                if isinstance(result, list) and len(result) > 0:
                    # Filter out our negative control label to check true positive confidence
                    positive_scores = [
                        item["score"] for item in result 
                        if item["label"] != "a random part of a computer desktop wallpaper or UI"
                    ]
                    confidence = max(positive_scores) if positive_scores else 0.0
                else:
                    confidence = 0.0
                
                print(f"✓ {confidence:.0%}")
                
                if confidence > best_score:
                    best_score = confidence
                    center_x = x1 + (crop_width // 2)
                    center_y = y1 + (crop_height // 2)
                    best_match = (center_x, center_y, confidence)
            
            except Exception as e:
                print(f"✗ {str(e)[:40]}")
                continue
        
        # Report results
        if best_match and best_score > 0.3:  # Confidence threshold
            x, y, conf = best_match
            print(f"\n✓ Grounding complete! Found '{target_app}' at ({x}, {y}) with {conf:.0%} confidence")
            return best_match
        else:
            print(f"\n⚠ OpenCLIP confidence too low ({best_score:.0%}), using heuristic fallback...")
            return self._heuristic_fallback(candidate_regions)
    
    @staticmethod
    def _heuristic_fallback(candidate_regions: List[List[int]]) -> Optional[Tuple[int, int, float]]:
        """
        Fallback: Return center of first (best-predicted) region.
        """
        if not candidate_regions:
            return None
        
        x1, y1, x2, y2 = candidate_regions[0]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        print(f"  Returning center of best-predicted region: ({center_x}, {center_y})")
        
        return (center_x, center_y, 0.5)


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
    ) -> Optional[Tuple[int, int, float]]:
        """Return center of first region."""
        if not candidate_regions:
            return None
        
        print(f"\n📍 Grounding phase (Heuristic): Using best-predicted region for '{target_app}'")
        
        x1, y1, x2, y2 = candidate_regions[0]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        print(f"  Result: ({center_x}, {center_y}) with 50% confidence (best region center)")
        
        return (center_x, center_y, 0.5)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from screenshot import take_screenshot
    from planner import GeminiPlanner, HeuristicPlanner
    
    print("=" * 70)
    print("FULL PIPELINE TEST: Planning (Heuristic) + Grounding (HF OpenCLIP)")
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
    except Exception as e:
        print(f"      DeepSeek failed: {e}")
        print("      Using heuristic planning...")
        planner = HeuristicPlanner()
        plan = planner.plan_icon_location(screenshot, target_app="Notepad")
        regions = plan["likely_regions"]
    
    # Step 3: Grounding phase
    print("\n[3/3] Grounding phase...")
    try:
        grounder = HFCLIPGrounder()
        result = grounder.ground_icon_in_regions(screenshot, regions, target_app="Notepad")
        
        if result:
            x, y, confidence = result
            print(f"\n✅ SUCCESS! Found Notepad at ({x}, {y}) with {confidence:.0%} confidence")
            print(f"   Ready to click and automate!")
        else:
            print("\n⚠ Grounding returned None")
    
    except Exception as e:
        print(f"\n❌ Grounding error: {e}")
        print("   Trying heuristic-only grounder...")
        try:
            grounder = SimpleHeuristicGrounder()
            result = grounder.ground_icon_in_regions(screenshot, regions, target_app="Notepad")
            if result:
                x, y, conf = result
                print(f"\n✅ Heuristic grounder: ({x}, {y}) with {conf:.0%} confidence")
        except Exception as e2:
            print(f"✗ All grounding methods failed: {e2}")
    
    print("\n" + "=" * 70)