"""
Verification Phase: Confirm that the selected cropped region contains the target element.
"""

import json
import os
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class QwenVerifier:
    """Precision verification system driven by Qwen2.5-VL."""
    
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

    def verify_target(self, cropped_image: Image.Image, instruction: str) -> Tuple[bool, str]:
        if not self.client:
            raise ValueError("OpenRouter client is uninitialized.")
            
        # Reject uninitialized or pitch-black crops immediately to avoid API loops
        extrema = cropped_image.convert("L").getextrema()
        if extrema == (0, 0):
            return False, instruction

        prompt = f"""You are a QA verification engine validating desktop user interface crops.
Task: Inspect this cropped visual element image patch and determine if it matches or contains the item: '{instruction}'.

Instructions:
1. Respond with "is_target" true only if the specified element is cleanly visible.
2. If the patch contains the icon but it is blurry, text-only, or off-center, you can modify 'refined_instruction'.
3. If it is a completely different element, or empty/black space, return false.
4. Output raw valid JSON matching the template below. No markdown wrapping.

Required JSON Structure:
{{
    "reasoning": "Visual proof analysis confirming identity.",
    "is_target": true,
    "refined_instruction": "{instruction}"
}}"""

        buffered = BytesIO()
        cropped_image.save(buffered, format="PNG")
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
        return bool(data.get("is_target", False)), str(data.get("refined_instruction", instruction))

class SimpleVerifier:
    """
    Simple heuristic verifier: just check if any element is visible.
    """
    
    @staticmethod
    def verify_target(
        cropped_screenshot: Image.Image,
        instruction: str
    ) -> Tuple[bool, str]:
        """
        Simple verification: assume positive if image is not empty.
        """
        # Simple heuristic: if image has any non-uniform pixels, assume target exists
        # In Python PIL, we can get standard extrema to see if there is color variation
        extrema = cropped_screenshot.convert("L").getextrema()
        if extrema and extrema[0] != extrema[1]:  # Min != Max pixel intensity
            return True, instruction
        else:
            return False, instruction


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from screenshot import take_screenshot
    
    print("=" * 70)
    print("VERIFIER TEST - Result Verification")
    print("=" * 70)
    
    # Capture a screenshot
    print("\n[1/2] Capturing desktop screenshot...")
    try:
        screenshot = take_screenshot()
        print(f"      Screenshot size: {screenshot.size}")
    except Exception as e:
        print(f"      Error: {e}")
        exit(1)
    
    # Create a test crop (assume we found something in top-left)
    print("\n[2/2] Testing Qwen Verifier...")
    crop = screenshot.crop((20, 240, 90, 320))
    
    try:
        verifier = QwenVerifier()
        is_target, new_instr = verifier.verify_target(crop, "Notepad application icon")
        
        print(f"\n✓ Verification Result:")
        print(f"  Is Target: {is_target}")
        print(f"  New Instruction: {new_instr}")
        
        assert isinstance(is_target, bool), "is_target must be boolean"
        print(f"\n✅ Verifier test passed!")
    
    except Exception as e:
        print(f"\n❌ Qwen verifier failed: {e}")
        print("  Trying heuristic verifier...")
        
        verifier = SimpleVerifier()
        is_target, new_instr = verifier.verify_target(crop, "Notepad application icon")
        print(f"\n✓ Heuristic: is_target={is_target}")
    
    print("\n" + "=" * 70)