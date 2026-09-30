"""
Preset Agricultural Sample Image Generator for Thesis Prototype.
Generates realistic 640x640 agricultural field images depicting challenging insect pest scenarios.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def _draw_natural_leaf_background(draw: ImageDraw.ImageDraw, width=640, height=640, base_color=(55, 110, 45)):
    # Draw gradient-like foliage background
    for y in range(0, height, 4):
        ratio = y / height
        r = int(base_color[0] + 15 * math.sin(ratio * math.pi))
        g = int(base_color[1] + 25 * math.cos(ratio * 2))
        b = int(base_color[2] + 10 * ratio)
        draw.rectangle([0, y, width, y + 4], fill=(max(0, r), max(0, g), max(0, b)))
        
    # Add diagonal foliage veins and shadows
    for x in range(-200, width + 200, 35):
        shade = (base_color[0] - 15, base_color[1] - 20, base_color[2] - 15)
        highlight = (base_color[0] + 20, base_color[1] + 30, base_color[2] + 15)
        draw.line([x, 0, x + 250, height], fill=shade, width=2)
        draw.line([x + 2, 0, x + 252, height], fill=highlight, width=1)

def _draw_rice_stems_background(draw: ImageDraw.ImageDraw, width=640, height=640):
    # Field mud / water base at bottom
    draw.rectangle([0, 0, width, height], fill=(70, 95, 55))
    
    # Draw vertical rice tillers/stems
    stems = [
        (120, 200, (110, 140, 70), (85, 115, 50)),
        (190, 280, (120, 150, 75), (95, 120, 55)),
        (270, 360, (115, 145, 70), (90, 118, 52)),
        (350, 440, (125, 155, 80), (100, 125, 58)),
        (430, 520, (110, 140, 68), (85, 112, 48)),
    ]
    for x1, x2, c_light, c_dark in stems:
        draw.rectangle([x1, 0, x2, height], fill=c_light)
        # Vertical striping on stems
        for sx in range(x1 + 6, x2, 8):
            draw.line([sx, 0, sx, height], fill=c_dark, width=2)

def _draw_planthopper(draw: ImageDraw.ImageDraw, cx, cy, size=40, angle=0):
    # Brown Planthopper: brownish-black body, translucent wings, distinctive delta shape
    body_col = (75, 45, 25)
    wing_col = (130, 90, 55)
    highlight = (165, 120, 80)
    
    # Body oval
    draw.ellipse([cx - size//3, cy - size//2, cx + size//3, cy + size//2], fill=body_col, outline=(40, 20, 10), width=2)
    # Head and eyes
    draw.ellipse([cx - size//4, cy - size//2 - 6, cx + size//4, cy - size//2 + 8], fill=(50, 30, 15))
    draw.ellipse([cx - size//4 - 2, cy - size//2, cx - size//4 + 3, cy - size//2 + 5], fill=(20, 10, 5))
    draw.ellipse([cx + size//4 - 3, cy - size//2, cx + size//4 + 2, cy - size//2 + 5], fill=(20, 10, 5))
    # Wings
    draw.polygon([
        (cx - size//3, cy - size//4),
        (cx + size//3, cy - size//4),
        (cx + size//2, cy + size//2 + 8),
        (cx, cy + size//2 + 14),
        (cx - size//2, cy + size//2 + 8)
    ], fill=wing_col, outline=highlight)
    # Antennae & legs
    draw.line([cx - size//3, cy, cx - size//2 - 6, cy - 8], fill=(45, 25, 10), width=2)
    draw.line([cx + size//3, cy, cx + size//2 + 6, cy - 8], fill=(45, 25, 10), width=2)
    draw.line([cx - size//3, cy + size//4, cx - size//2 - 8, cy + size//4 + 10], fill=(45, 25, 10), width=2)
    draw.line([cx + size//3, cy + size//4, cx + size//2 + 8, cy + size//4 + 10], fill=(45, 25, 10), width=2)

def _draw_leaf_roller(draw: ImageDraw.ImageDraw, x1, y1, x2, y2):
    # Rice leaf roller moth: yellowish-brown triangular moth with dark wavy transverse bands
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = (x2 - x1)
    h = (y2 - y1)
    
    # Delta-shaped moth silhouette
    p1 = (cx, y1 + 10)
    p2 = (x1 + 10, y2 - 10)
    p3 = (x2 - 10, y2 - 10)
    draw.polygon([p1, p2, p3], fill=(185, 140, 75), outline=(90, 60, 25), width=2)
    
    # Wavy transverse wing bands
    draw.arc([x1 + 15, cy - 20, x2 - 15, cy + 20], start=0, end=180, fill=(100, 65, 30), width=3)
    draw.arc([x1 + 25, cy + 5, x2 - 25, cy + 45], start=0, end=180, fill=(100, 65, 30), width=3)
    
    # Fringed margin at bottom
    draw.line([x1 + 10, y2 - 10, x2 - 10, y2 - 10], fill=(80, 50, 20), width=4)

def _draw_dried_leaf_tip(draw: ImageDraw.ImageDraw, x1, y1, x2, y2):
    # Dried curled leaf tip that confuses baseline detector (False Positive)
    draw.polygon([
        (x1 + 5, y1 + 10),
        (x2 - 15, y1 + 30),
        (x2 - 5, y2 - 10),
        (x1 + 20, y2 - 5)
    ], fill=(160, 125, 65), outline=(110, 85, 40), width=2)
    draw.line([x1 + 10, y1 + 20, x2 - 10, y2 - 10], fill=(95, 70, 30), width=2)

def _draw_aphid(draw: ImageDraw.ImageDraw, cx, cy, size=24):
    # Pear-shaped greenish-yellow aphid with cornicles
    draw.ellipse([cx - size//2, cy - size//2, cx + size//2, cy + size//2], fill=(170, 195, 55), outline=(95, 120, 25), width=1)
    # Head
    draw.ellipse([cx - size//4, cy - size//2 - 3, cx + size//4, cy - size//2 + 5], fill=(140, 165, 40))
    # Cornicles (tail tubes)
    draw.line([cx - size//4, cy + size//3, cx - size//3 - 3, cy + size//2 + 2], fill=(80, 100, 20), width=1)
    draw.line([cx + size//4, cy + size//3, cx + size//3 + 3, cy + size//2 + 2], fill=(80, 100, 20), width=1)

def _draw_corn_borer(draw: ImageDraw.ImageDraw, x1, y1, x2, y2):
    # Asiatic corn borer caterpillar: segmented yellowish-pink body with dark pinacula spots
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    
    # Body curve segments
    points = [
        (x1 + 20, y1 + 30),
        (cx - 20, cy - 20),
        (cx + 20, cy + 10),
        (x2 - 30, y2 - 25)
    ]
    # Draw segments
    for i, (px, py) in enumerate(points):
        rad = 22 - i * 2
        draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(225, 200, 170), outline=(130, 95, 65), width=2)
        # Pinacula (dark spots on dorsal side)
        draw.ellipse([px - 5, py - rad + 3, px - 2, py - rad + 6], fill=(70, 45, 25))
        draw.ellipse([px + 2, py - rad + 3, px + 5, py - rad + 6], fill=(70, 45, 25))
        
    # Dark brown head capsule
    hx, hy = points[0]
    draw.ellipse([hx - 16, hy - 16, hx + 16, hy + 16], fill=(110, 50, 20), outline=(60, 25, 10), width=2)

def _draw_plant_bug(draw: ImageDraw.ImageDraw, x1, y1, x2, y2):
    # Shield bug / stink bug shape
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Shield polygon
    draw.polygon([
        (cx, y1 + 15),
        (x2 - 15, cy - 10),
        (x2 - 25, y2 - 20),
        (cx, y2 - 5),
        (x1 + 25, y2 - 20),
        (x1 + 15, cy - 10)
    ], fill=(90, 130, 95), outline=(40, 70, 45), width=2)
    
    # Pronotum & scutellum triangle
    draw.polygon([
        (cx - 25, cy - 10),
        (cx + 25, cy - 10),
        (cx, cy + 25)
    ], fill=(120, 160, 110), outline=(50, 80, 50))


def generate_preset_images(output_dir="assets/presets") -> dict:
    """
    Generates all 4 preset agricultural pest images and saves them to output_dir.
    Returns: dict mapping case_id -> absolute file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = {}
    
    # --- Case 1: Clustered Infestation (Brown Planthopper) ---
    img1 = Image.new("RGB", (640, 640), color=(60, 85, 45))
    d1 = ImageDraw.Draw(img1)
    _draw_rice_stems_background(d1, 640, 640)
    # Draw clustered planthoppers at specific coordinates matching config
    _draw_planthopper(d1, cx=215, cy=260, size=46)
    _draw_planthopper(d1, cx=265, cy=292, size=44)
    _draw_planthopper(d1, cx=380, cy=335, size=48)
    _draw_planthopper(d1, cx=415, cy=352, size=40)
    # Slight blur then sharpen for photographic depth
    img1 = img1.filter(ImageFilter.GaussianBlur(radius=0.5))
    p1 = os.path.join(output_dir, "clustered_bph.jpg")
    img1.save(p1, quality=95)
    generated["clustered_bph"] = p1
    
    # --- Case 2: Camouflaged Pest (Rice Leaf Roller) ---
    img2 = Image.new("RGB", (640, 640), color=(50, 105, 40))
    d2 = ImageDraw.Draw(img2)
    _draw_natural_leaf_background(d2, 640, 640, base_color=(55, 115, 42))
    # Draw pest
    _draw_leaf_roller(d2, x1=250, y1=200, x2=395, y2=355)
    # Draw dried leaf tip (baseline false positive artifact)
    _draw_dried_leaf_tip(d2, x1=475, y1=375, x2=555, y2=455)
    img2 = img2.filter(ImageFilter.GaussianBlur(radius=0.4))
    p2 = os.path.join(output_dir, "camouflaged_leafroller.jpg")
    img2.save(p2, quality=95)
    generated["camouflaged_leafroller"] = p2
    
    # --- Case 3: Micro-scale Pest (Aphids) ---
    img3 = Image.new("RGB", (640, 640), color=(45, 95, 38))
    d3 = ImageDraw.Draw(img3)
    _draw_natural_leaf_background(d3, 640, 640, base_color=(45, 100, 38))
    # Micro aphids
    _draw_aphid(d3, cx=235, cy=185, size=28)
    _draw_aphid(d3, cx=345, cy=295, size=26)
    _draw_aphid(d3, cx=437, cy=387, size=24)
    img3 = img3.filter(ImageFilter.GaussianBlur(radius=0.3))
    p3 = os.path.join(output_dir, "micro_aphids.jpg")
    img3.save(p3, quality=95)
    generated["micro_aphids"] = p3
    
    # --- Case 4: Multi-Class Field Infestation ---
    img4 = Image.new("RGB", (640, 640), color=(50, 90, 40))
    d4 = ImageDraw.Draw(img4)
    _draw_natural_leaf_background(d4, 640, 640, base_color=(58, 108, 48))
    _draw_corn_borer(d4, x1=140, y1=180, x2=285, y2=335)
    _draw_plant_bug(d4, x1=360, y1=260, x2=485, y2=405)
    img4 = img4.filter(ImageFilter.GaussianBlur(radius=0.4))
    p4 = os.path.join(output_dir, "multiclass_field.jpg")
    img4.save(p4, quality=95)
    generated["multiclass_field"] = p4
    
    return generated

if __name__ == "__main__":
    paths = generate_preset_images()
    print("Generated preset images successfully:", paths)
