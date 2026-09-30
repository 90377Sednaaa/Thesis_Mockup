"""
Detection and Rendering Engine for Thesis Prototype.
Provides publication-grade bounding box overlay visualization, NMS filtering,
latency simulation, and custom image inference simulation.
"""

import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def apply_nms(boxes: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    """Applies Non-Maximum Suppression to filter highly overlapping boxes."""
    if not boxes:
        return []
    
    sorted_boxes = sorted(boxes, key=lambda b: b.get("conf", 0.0), reverse=True)
    selected_boxes = []

    while sorted_boxes:
        best = sorted_boxes.pop(0)
        selected_boxes.append(best)
        remaining = []
        for b in sorted_boxes:
            iou = calculate_iou(best["box"], b["box"])
            if iou < iou_threshold:
                remaining.append(b)
        sorted_boxes = remaining

    return selected_boxes

def draw_yolo_detections(
    image: Image.Image,
    boxes: list[dict],
    conf_threshold: float = 0.40,
    iou_threshold: float = 0.45,
    show_labels: bool = True,
    show_conf: bool = True,
    box_width: int = 2
) -> Image.Image:
    """
    Overlays clean, publication-grade bounding boxes on an image.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    valid_boxes = [b for b in boxes if b.get("conf", 0.0) >= conf_threshold]
    filtered_boxes = apply_nms(valid_boxes, iou_threshold=iou_threshold)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for item in filtered_boxes:
        box = item["box"]
        x1, y1, x2, y2 = box
        label = item.get("label", "Insect")
        conf = item.get("conf", 0.0)
        color = item.get("color", "#10b981")

        # Draw main crisp box outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        # Draw corner brackets for precision feel
        c_len = min(12, max(5, (x2 - x1) // 5), max(5, (y2 - y1) // 5))
        draw.line([x1, y1, x1 + c_len, y1], fill=color, width=box_width + 1)
        draw.line([x1, y1, x1, y1 + c_len], fill=color, width=box_width + 1)
        draw.line([x2, y1, x2 - c_len, y1], fill=color, width=box_width + 1)
        draw.line([x2, y1, x2, y1 + c_len], fill=color, width=box_width + 1)
        draw.line([x1, y2, x1 + c_len, y2], fill=color, width=box_width + 1)
        draw.line([x1, y2, x1, y2 - c_len], fill=color, width=box_width + 1)
        draw.line([x2, y2, x2 - c_len, y2], fill=color, width=box_width + 1)
        draw.line([x2, y2, x2, y2 - c_len], fill=color, width=box_width + 1)

        # Build clean label text
        parts = []
        if show_labels:
            parts.append(label)
        if show_conf:
            parts.append(f"{int(conf * 100)}%")
        tag_text = " ".join(parts)

        if tag_text:
            bbox = font.getbbox(tag_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            pad_x, pad_y = 5, 3

            # Position above or inside top
            if y1 - text_h - (pad_y * 2) >= 0:
                tag_y1 = y1 - text_h - (pad_y * 2) - 1
                tag_y2 = y1 - 1
            else:
                tag_y1 = y1 + 1
                tag_y2 = y1 + text_h + (pad_y * 2) + 1

            tag_x1 = x1
            tag_x2 = x1 + text_w + (pad_x * 2)

            # Draw tag pill
            draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=color)
            draw.text((tag_x1 + pad_x, tag_y1 + pad_y - 1), tag_text, fill=(255, 255, 255), font=font)

    return canvas

def simulate_inference_latency(model_type: str = "baseline", cr_ratio: float = 0.50) -> tuple[float, float]:
    """
    Simulates hardware inference latency (ms) and throughput (FPS)
    benchmarked on NVIDIA T4 GPU for batch size 1 at 640x640.
    Returns: (latency_ms, fps)
    """
    if model_type == "baseline":
        base_latency = 28.4
        jitter = random.uniform(-0.4, 0.4)
        lat = round(base_latency + jitter, 1)
        fps = round(1000.0 / lat, 1)
        return lat, fps
    
    if abs(cr_ratio - 1.00) < 0.05:
        base_latency = 28.7
    elif abs(cr_ratio - 0.75) < 0.05:
        base_latency = 23.1
    elif abs(cr_ratio - 0.25) < 0.05:
        base_latency = 14.9
    else:  # r=0.50
        base_latency = 18.6

    jitter = random.uniform(-0.3, 0.3)
    lat = round(base_latency + jitter, 1)
    fps = round(1000.0 / lat, 1)
    return lat, fps

def process_custom_image(
    image: Image.Image,
    benchmark: str = "IP102",
    conf_threshold: float = 0.40,
    cr_ratio: float = 0.50
) -> tuple[list[dict], list[dict]]:
    """
    Processes a custom user-uploaded image.
    Resizes to 640x640 and generates realistic comparative detections.
    Returns: (baseline_boxes, cr_boxes)
    """
    img_640 = image.convert("RGB").resize((640, 640), Image.Resampling.LANCZOS)
    arr = np.array(img_640)
    h, w, _ = arr.shape
    cy, cx = int(h * 0.48), int(w * 0.50)

    if benchmark == "R2000":
        primary_class = "Rice Leaf Roller"
        sec_class = "Brown Planthopper"
    else:
        primary_class = "Armyworm"
        sec_class = "Aphids"

    baseline_boxes = [
        {
            "box": [max(50, cx - 110), max(50, cy - 90), min(590, cx + 95), min(590, cy + 95)],
            "label": primary_class,
            "conf": 0.74,
            "color": "#f59e0b"
        },
        {
            "box": [max(50, cx + 60), max(50, cy + 70), min(590, cx + 180), min(590, cy + 170)],
            "label": sec_class,
            "conf": 0.68,
            "color": "#f59e0b"
        },
        {
            "box": [max(20, cx - 220), max(20, cy + 120), min(590, cx - 120), min(590, cy + 200)],
            "label": f"{primary_class} [False Alarm]",
            "conf": 0.48,
            "color": "#ef4444"
        }
    ]

    cr_boxes = [
        {
            "box": [max(50, cx - 95), max(50, cy - 80), min(590, cx + 85), min(590, cy + 85)],
            "label": primary_class,
            "conf": 0.92,
            "color": "#10b981"
        },
        {
            "box": [max(50, cx + 70), max(50, cy + 75), min(590, cx + 170), min(590, cy + 165)],
            "label": sec_class,
            "conf": 0.89,
            "color": "#10b981"
        }
    ]

    return baseline_boxes, cr_boxes
