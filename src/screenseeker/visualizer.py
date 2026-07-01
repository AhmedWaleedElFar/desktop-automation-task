import os
from typing import Tuple, List
from PIL import Image, ImageDraw, ImageFont

def save_annotated_image(base_img: Image.Image, boxes: List[List[int]], output_path: str):
    """Save an image with simple red bounding boxes for debugging."""
    annotated = base_img.copy()
    draw = ImageDraw.Draw(annotated)
    for i, box in enumerate(boxes):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0] + 2, box[1] + 2), f"#{i}", fill="red")
    annotated.save(output_path)


def save_detection_screenshot(
    screenshot: Image.Image,
    center: Tuple[int, int],
    bounding_box: Tuple[int, int, int, int],
    confidence: float,
    run_timestamp: str,
    output_dir: str,
    label: str = "Target"
) -> str:
    """Save a richly annotated detection screenshot for submission.

    Draws:
    - Green bounding box around the detected icon
    - Crosshair lines through the center
    - Filled circle at the center point
    - Coordinate + confidence text overlay
    - Timestamp watermark

    Returns the saved file path.
    """
    annotated = screenshot.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    cx, cy = center
    x1, y1, x2, y2 = bounding_box

    # --- bounding box ---
    draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 80), width=3)

    # --- crosshair ---
    line_len = 30
    draw.line([(cx - line_len, cy), (cx + line_len, cy)], fill=(0, 220, 80), width=2)
    draw.line([(cx, cy - line_len), (cx, cy + line_len)], fill=(0, 220, 80), width=2)

    # --- center dot ---
    r = 6
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 220, 80))

    # --- text label background + text ---
    tag = f"  {label}  ({cx}, {cy})  conf={confidence:.0%}  "
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    tx, ty = x1, max(0, y1 - 26)
    bbox_t = draw.textbbox((tx, ty), tag, font=font)
    draw.rectangle(bbox_t, fill=(0, 160, 60))
    draw.text((tx, ty), tag, fill=(255, 255, 255), font=font)

    # --- timestamp watermark (bottom-left) ---
    ts_text = f"  Detected: {run_timestamp.replace('_', ' ')}  "
    img_w, img_h = annotated.size
    try:
        ts_font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        ts_font = ImageFont.load_default()
    ts_bbox = draw.textbbox((4, img_h - 24), ts_text, font=ts_font)
    draw.rectangle(ts_bbox, fill=(0, 0, 0))
    draw.text((4, img_h - 24), ts_text, fill=(200, 255, 200), font=ts_font)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"{run_timestamp}_detection.png"
    )
    annotated.save(out_path)
    print(f"📸 Annotated detection screenshot saved: {out_path}")
    return out_path
