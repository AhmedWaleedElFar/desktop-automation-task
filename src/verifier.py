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
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠ Warning: anthropic not installed.")


class ClaudeVerifier:
    """
    Verification phase using Claude API to verify if detected element matches instruction.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude verifier.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic not installed. Run: uv pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    
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
        
        base64_image = self._image_to_base64(cropped_screenshot)
        
        prompt = f"""You are a visual verification expert.

Instruction: {instruction}

Look at this cropped screenshot. Your task is to determine if it contains the target described in the instruction.

Analyze the screenshot and respond with ONLY this JSON (no code blocks or markdown):
{{
    "result": "is_target" | "target_elsewhere" | "target_not_found",
    "reasoning": "Brief explanation of your analysis",
    "new_instruction": "A clearer/more specific version of the instruction if needed (null if not needed)"
}}

result values:
- "is_target": The cropped region contains the target element
- "target_elsewhere": The target exists in the full screenshot but not in this crop
- "target_not_found": The target does not appear to exist

Return ONLY the JSON object, nothing else."""
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
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
            
            response_text = response.content[0].text.strip()
            
            # Remove markdown
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            # Validate
            valid_results = ["is_target", "target_elsewhere", "target_not_found"]
            if result["result"] not in valid_results:
                raise ValueError(f"Invalid result: {result['result']}")
            
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
        
        Args:
            cropped_screenshot: PIL Image of the detected region
            instruction: Target description
        
        Returns:
            Tuple of (is_target, instruction)
        """
        # Simple heuristic: if image has any non-uniform pixels, assume target exists
        pixels = list(cropped_screenshot.get_flattened_data())
        
        if len(set(pixels)) > 1:  # More than one unique color
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
    print("\n[2/2] Testing Claude Verifier...")
    crop = screenshot.crop((0, 0, 100, 100))
    
    try:
        verifier = ClaudeVerifier()
        is_target, new_instr = verifier.verify_target(crop, "Notepad application icon")
        
        print(f"\n✓ Verification Result:")
        print(f"  Is Target: {is_target}")
        print(f"  New Instruction: {new_instr}")
        
        assert isinstance(is_target, bool), "is_target must be boolean"
        print(f"\n✅ Verifier test passed!")
    
    except Exception as e:
        print(f"\n❌ Claude verifier failed: {e}")
        print("  Trying heuristic verifier...")
        
        verifier = SimpleVerifier()
        is_target, new_instr = verifier.verify_target(crop, "Notepad application icon")
        print(f"\n✓ Heuristic: is_target={is_target}")
    
    print("\n" + "=" * 70)
