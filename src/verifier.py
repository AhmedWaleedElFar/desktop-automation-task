"""
Verification Phase: Confirm that detected element matches the instruction.
Inspired by ScreenSeekeR's result verification step.

After grounding, ask the planner: "Is this actually the target?"
"""

import json
import os
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ Warning: google-genai not installed.")


class GeminiVerifier:
    """
    Verification phase using Gemini API to verify if detected element matches instruction.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini verifier.
        
        Args:
            api_key: Google Gemini API key. If None, uses GEMINI_API_KEY env var.
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        
        self.client = genai.Client(api_key=self.api_key)
    
    def verify_target(
        self,
        cropped_screenshot: Image.Image,
        instruction: str
    ) -> Tuple[bool, str]:
        """
        Verify if the cropped screenshot contains the target element.
        
        Args:
            cropped_screenshot: PIL Image of the detected region (cropped)
            instruction: Target description (e.g., "Notepad icon")
        
        Returns:
            Tuple of (is_target: bool, refined_instruction: str)
            - is_target: True if this is the target, False otherwise
            - refined_instruction: Clearer version of instruction if target not found
        """
        
        prompt = f"""You are a visual verification expert.

Instruction: {instruction}

Look at this cropped screenshot. Your task is to determine if it contains the target described in the instruction.

Analyze the screenshot and respond with ONLY this JSON. Do not output markdown code blocks.

Schema:
{{
    "result": "is_target" | "target_elsewhere" | "target_not_found",
    "reasoning": "Brief explanation of your analysis",
    "new_instruction": "A clearer/more specific version of the instruction if needed (null if not needed)"
}}

result values:
- "is_target": The cropped region contains the target element
- "target_elsewhere": The target exists in the full screenshot but not in this crop
- "target_not_found": The target does not appear to exist"""
        
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[cropped_screenshot, prompt],
                config=config
            )
            
            response_text = response.text.strip()
            result = json.loads(response_text)
            
            # Validate output keys and values
            valid_results = ["is_target", "target_elsewhere", "target_not_found"]
            if "result" not in result or result["result"] not in valid_results:
                raise ValueError(f"Invalid result: {result.get('result')}")
            
            is_target = (result["result"] == "is_target")
            new_instruction = result.get("new_instruction") or instruction
            
            print(f"✓ Verification: {result['result']}")
            print(f"  Reasoning: {result['reasoning'][:80]}...")
            
            return is_target, new_instruction
        
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse verifier response as JSON: {e}")
            print(f"Response: {response_text[:300]}")
            raise
        except Exception as e:
            print(f"✗ Verification error: {e}")
            raise


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
    print("\n[2/2] Testing Gemini Verifier...")
    crop = screenshot.crop((20, 240, 90, 320))
    
    try:
        verifier = GeminiVerifier()
        is_target, new_instr = verifier.verify_target(crop, "Notepad application icon")
        
        print(f"\n✓ Verification Result:")
        print(f"  Is Target: {is_target}")
        print(f"  New Instruction: {new_instr}")
        
        assert isinstance(is_target, bool), "is_target must be boolean"
        print(f"\n✅ Verifier test passed!")
    
    except Exception as e:
        print(f"\n❌ Gemini verifier failed: {e}")
        print("  Trying heuristic verifier...")
        
        verifier = SimpleVerifier()
        is_target, new_instr = verifier.verify_target(crop, "Notepad application icon")
        print(f"\n✓ Heuristic: is_target={is_target}")
    
    print("\n" + "=" * 70)