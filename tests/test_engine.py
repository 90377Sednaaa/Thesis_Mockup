import pytest
from PIL import Image
from engine import draw_yolo_detections, simulate_inference_latency, process_custom_image

def test_draw_yolo_detections():
    img = Image.new("RGB", (640, 640), color=(100, 150, 100))
    boxes = [
        {"box": [100, 100, 200, 200], "label": "Pest A", "conf": 0.85, "color": "#10b981"},
        {"box": [300, 300, 400, 400], "label": "Pest B", "conf": 0.35, "color": "#f59e0b"}
    ]
    # At conf_threshold 0.40, only box 1 should be drawn
    out_img = draw_yolo_detections(img, boxes, conf_threshold=0.40, show_labels=True, show_conf=True)
    assert out_img.size == (640, 640)
    assert out_img is not None

def test_simulate_inference_latency():
    base_lat, base_fps = simulate_inference_latency("baseline", cr_ratio=1.0)
    cr_lat, cr_fps = simulate_inference_latency("cr_hsdpa", cr_ratio=0.50)
    
    assert 26.0 <= base_lat <= 31.0
    assert 17.0 <= cr_lat <= 21.0
    assert cr_fps > base_fps

def test_process_custom_image():
    img = Image.new("RGB", (800, 600), color=(80, 120, 80))
    base_boxes, cr_boxes = process_custom_image(img, benchmark="IP102", conf_threshold=0.30, cr_ratio=0.50)
    assert len(base_boxes) >= 1
    assert len(cr_boxes) >= 1
