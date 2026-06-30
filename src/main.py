"""
Diagnostic script to test HF OpenCLIP API and find working models.
"""
import requests
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("❌ HF_TOKEN not set. Add to .env file")
    exit(1)

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Generate a simple test image
test_img = Image.new('RGB', (100, 100), color='red')
buffer = BytesIO()
test_img.save(buffer, format="PNG")
base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

print("Testing HF Vision Models...\n")

# Test different model endpoints and task types
models_to_test = [
    {
        "name": "CLIP (Zero-Shot Classification)",
        "url": "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32",
        "task": "zero-shot-image-classification",
        "payload": {
            "image": base64_img,
            "parameters": {"candidate_labels": ["a notepad icon", "desktop"]}
        }
    },
    {
        "name": "Florence-2 (Dense Region Caption)",
        "url": "https://api-inference.huggingface.co/models/microsoft/Florence-2-large",
        "task": "object-detection",
        "payload": {
            "image": base64_img,
            "parameters": {"task": "region_classification", "text": "notepad icon"}
        }
    },
    {
        "name": "Qwen2-VL (Visual QA)",
        "url": "https://api-inference.huggingface.co/models/Qwen/Qwen2-VL-7B-Instruct",
        "task": "visual-question-answering",
        "payload": {
            "image": base64_img,
            "question": "Is there a notepad icon in this image?"
        }
    },
    {
        "name": "CLIP (via direct model)",
        "url": "https://api-inference.huggingface.co/models/laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "task": "feature-extraction",
        "payload": {
            "image": base64_img,
            "inputs": "a notepad icon"
        }
    }
]

for model_test in models_to_test:
    print(f"🧪 Testing: {model_test['name']}")
    print(f"   URL: {model_test['url']}")
    
    try:
        response = requests.post(
            model_test['url'],
            headers=headers,
            json=model_test['payload'],
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"   ✅ SUCCESS ({response.status_code})")
            print(f"   Response type: {type(response.json())}")
            print(f"   Preview: {str(response.json())[:100]}...\n")
        else:
            print(f"   ❌ FAILED ({response.status_code})")
            print(f"   Error: {response.text[:150]}...\n")
    
    except requests.exceptions.Timeout:
        print(f"   ⏱️ TIMEOUT (10s)\n")
    except requests.exceptions.ConnectionError as e:
        print(f"   🔌 CONNECTION ERROR: {str(e)[:80]}...\n")
    except Exception as e:
        print(f"   ❌ EXCEPTION: {type(e).__name__}: {str(e)[:80]}...\n")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
print("""
If CLIP (zero-shot) failed but another model worked:
  → Update grounder.py to use that model's endpoint

If ALL failed:
  → Network/token issue. Check HF_TOKEN and internet connection.
  → Try using LOCAL models instead (Ollama + LLaVA)

If you want vision + structured output:
  → Use Claude API (via Artifacts feature) with vision
  → Or local Ollama + LLaVA with custom prompting
""")