# Thesis Insect Pest Detection Mockup & Evaluation Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a defense-ready Streamlit prototype that performs side-by-side visual and computational comparisons between baseline Attention-PestNet and proposed Modified Attention-PestNet (CR-HSDPA) on IP102 and R2000 insect pest benchmarks.

**Architecture:** A modular Python/Streamlit application featuring a dedicated configuration store for empirical benchmark data, a sample generator for representative agricultural test cases, an image rendering engine that overlays YOLO-style bounding boxes, and an interactive dashboard with side-by-side visualizers, live KPI metric delta badges, and an architectural defense explainer.

**Tech Stack:** Python 3.10+, Streamlit, Pillow (PIL), NumPy, Matplotlib, Pytest.

## Global Constraints
- Pure mockup / simulation engine: Zero external GPU/CUDA dependency so it runs instantaneously and reliably during defense presentations.
- Resolution standard: 640×640 standard input resolution matching the thesis experimental setup.
- Metrics fidelity: Exact reproduction of baseline Attention-PestNet (79.66M params, 162.7 GFLOPs, 68.67% mAP@50 on IP102) and thesis CR-HSDPA (58.2M params, 118.4 GFLOPs, 69.40% mAP@50 on IP102).
- Clean code architecture: Modular components (`config.py`, `sample_generator.py`, `engine.py`, `components.py`, `app.py`).

---

### Task 1: Configuration & Benchmark Data Store

**Files:**
- Create: `d:/Thesisss/Thesis_Mockup/config.py`
- Test: `d:/Thesisss/Thesis_Mockup/tests/test_config.py`

**Interfaces:**
- Produces:
  - `BENCHMARKS`: Dictionary containing IP102 and R2000 baseline vs. CR-HSDPA metrics (mAP@50, mAP@50:95, Params, GFLOPs, Model Size, Latency, FPS).
  - `ABLATION_DATA`: Dictionary containing ablation experiments ($r = 1.00, 0.75, 0.50, 0.25$).
  - `PRESET_CASES`: Metadata defining the 4 preset agricultural challenge scenarios.
  - `CLASS_COLORS`: Hex color palette for insect pest classes.

- [ ] **Step 1: Write the failing test**
Create `tests/test_config.py`:
```python
import pytest
from config import BENCHMARKS, ABLATION_DATA, PRESET_CASES

def test_benchmarks_exist_and_contain_required_metrics():
    assert "IP102" in BENCHMARKS
    assert "R2000" in BENCHMARKS
    
    for bm in ["IP102", "R2000"]:
        data = BENCHMARKS[bm]
        assert "baseline" in data
        assert "cr_hsdpa" in data
        
        # Verify baseline metrics
        b = data["baseline"]
        c = data["cr_hsdpa"]
        assert b["map50"] < c["map50"]
        assert b["gflops"] > c["gflops"]
        assert b["params_m"] > c["params_m"]
        assert b["latency_ms"] > c["latency_ms"]
        assert b["fps"] < c["fps"]

def test_ablation_data_contains_all_ratios():
    ratios = [row["variant"] for row in ABLATION_DATA]
    assert "Baseline HSDPA" in ratios
    assert "CR-1.00" in ratios
    assert "CR-0.75" in ratios
    assert "CR-0.50" in ratios
    assert "CR-0.25" in ratios

def test_preset_cases_defined():
    assert len(PRESET_CASES) == 4
    case_ids = [c["id"] for c in PRESET_CASES]
    assert "clustered_bph" in case_ids
    assert "camouflaged_leafroller" in case_ids
    assert "micro_aphids" in case_ids
    assert "multiclass_field" in case_ids
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_config.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Implement `config.py`**
Create `config.py` with all benchmark metrics, ablation tables, class color mappings, and preset challenge case definitions:
```python
"""
Configuration and empirical benchmark metrics for the thesis:
'A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention for Efficient Insect Pest Detection'
"""

BENCHMARKS = {
    "IP102": {
        "title": "IP102 (General Insect Pest Benchmark - 102 Classes)",
        "num_classes": 102,
        "input_size": "640x640",
        "hardware": "NVIDIA T4 GPU (FP32, Batch 1)",
        "baseline": {
            "name": "Attention-PestNet (Baseline)",
            "architecture": "SDC + MSPA + 3×HSDPA (Full Channels C)",
            "map50": 68.67,
            "map50_95": 44.17,
            "params_m": 79.66,
            "gflops": 162.70,
            "model_size_mb": 152.3,
            "latency_ms": 28.4,
            "fps": 35.2,
        },
        "cr_hsdpa": {
            "name": "Modified Attention-PestNet (CR-HSDPA, r=0.50)",
            "architecture": "SDC + MSPA + 3×CR-HSDPA (Bottleneck 1×1, Cr=0.5C)",
            "map50": 69.40,
            "map50_95": 44.80,
            "params_m": 58.20,
            "gflops": 118.40,
            "model_size_mb": 112.5,
            "latency_ms": 18.6,
            "fps": 53.8,
        }
    },
    "R2000": {
        "title": "R2000 (Rice-Specific Pest Benchmark - 16 Classes)",
        "num_classes": 16,
        "input_size": "640x640",
        "hardware": "NVIDIA T4 GPU (FP32, Batch 1)",
        "baseline": {
            "name": "Attention-PestNet (Baseline)",
            "architecture": "SDC + MSPA + 3×HSDPA (Full Channels C)",
            "map50": 82.60,
            "map50_95": 68.20,
            "params_m": 79.66,
            "gflops": 162.70,
            "model_size_mb": 152.3,
            "latency_ms": 28.1,
            "fps": 35.6,
        },
        "cr_hsdpa": {
            "name": "Modified Attention-PestNet (CR-HSDPA, r=0.50)",
            "architecture": "SDC + MSPA + 3×CR-HSDPA (Bottleneck 1×1, Cr=0.5C)",
            "map50": 83.30,
            "map50_95": 68.90,
            "params_m": 58.20,
            "gflops": 118.40,
            "model_size_mb": 112.5,
            "latency_ms": 18.4,
            "fps": 54.3,
        }
    }
}

ABLATION_DATA = [
    {
        "variant": "Baseline HSDPA",
        "ratio": "1.00 (None)",
        "bottleneck": "No",
        "tda_levels": 4,
        "params_m": 79.66,
        "gflops": 162.70,
        "delta_gflops": "0.0%",
        "map50": 68.67,
        "map50_95": 44.17,
        "latency_ms": 28.4,
        "fps": 35.2,
        "status": "Baseline Reference"
    },
    {
        "variant": "CR-1.00",
        "ratio": "1.00",
        "bottleneck": "Yes (Control)",
        "tda_levels": 4,
        "params_m": 79.82,
        "gflops": 163.10,
        "delta_gflops": "+0.2%",
        "map50": 68.72,
        "map50_95": 44.21,
        "latency_ms": 28.7,
        "fps": 34.8,
        "status": "Architectural Control"
    },
    {
        "variant": "CR-0.75",
        "ratio": "0.75",
        "bottleneck": "Yes",
        "tda_levels": 4,
        "params_m": 68.50,
        "gflops": 139.20,
        "delta_gflops": "-14.4%",
        "map50": 69.10,
        "map50_95": 44.50,
        "latency_ms": 23.1,
        "fps": 43.3,
        "status": "Mild Compression"
    },
    {
        "variant": "CR-0.50",
        "ratio": "0.50",
        "bottleneck": "Yes",
        "tda_levels": 4,
        "params_m": 58.20,
        "gflops": 118.40,
        "delta_gflops": "-27.2%",
        "map50": 69.40,
        "map50_95": 44.80,
        "latency_ms": 18.6,
        "fps": 53.8,
        "status": "Selected Primary (Optimal Pareto)"
    },
    {
        "variant": "CR-0.25",
        "ratio": "0.25",
        "bottleneck": "Yes",
        "tda_levels": 4,
        "params_m": 48.00,
        "gflops": 97.60,
        "delta_gflops": "-40.0%",
        "map50": 67.80,
        "map50_95": 43.40,
        "latency_ms": 14.9,
        "fps": 67.1,
        "status": "Aggressive (Feature Loss)"
    }
]

PRESET_CASES = [
    {
        "id": "clustered_bph",
        "title": "Case 1: Dense Clustered Infestation",
        "pest_name": "Brown Planthopper (Nilaparvata lugens)",
        "benchmark": "R2000",
        "challenge": "Multiple overlapping small insects at rice stem base.",
        "description": "Baseline produces overlapping, lower-confidence bounding boxes; CR-HSDPA resolves individual pests cleanly with higher confidence.",
        "baseline_boxes": [
            {"box": [180, 220, 260, 310], "label": "Brown Planthopper", "conf": 0.73, "color": "#f59e0b"},
            {"box": [220, 250, 305, 340], "label": "Brown Planthopper", "conf": 0.69, "color": "#f59e0b"},
            {"box": [340, 290, 420, 380], "label": "Brown Planthopper", "conf": 0.77, "color": "#f59e0b"},
            {"box": [380, 310, 450, 395], "label": "Brown Planthopper", "conf": 0.64, "color": "#f59e0b"}
        ],
        "cr_boxes": [
            {"box": [182, 222, 255, 308], "label": "Brown Planthopper", "conf": 0.89, "color": "#10b981"},
            {"box": [235, 255, 300, 335], "label": "Brown Planthopper", "conf": 0.86, "color": "#10b981"},
            {"box": [342, 292, 418, 378], "label": "Brown Planthopper", "conf": 0.92, "color": "#10b981"},
            {"box": [382, 315, 448, 390], "label": "Brown Planthopper", "conf": 0.84, "color": "#10b981"}
        ]
    },
    {
        "id": "camouflaged_leafroller",
        "title": "Case 2: Camouflaged Pest on Foliage",
        "pest_name": "Rice Leaf Roller (Cnaphalocrocis medinalis)",
        "benchmark": "R2000",
        "challenge": "Pest coloration and texture blend into rice leaf veins.",
        "description": "Baseline produces a false positive on curled dried leaf tip; CR-HSDPA's channel-reduced attention suppresses background noise and accurately detects only the pest.",
        "baseline_boxes": [
            {"box": [250, 200, 400, 360], "label": "Rice Leaf Roller", "conf": 0.75, "color": "#f59e0b"},
            {"box": [480, 380, 560, 460], "label": "Rice Leaf Roller [FP]", "conf": 0.52, "color": "#ef4444"}
        ],
        "cr_boxes": [
            {"box": [255, 205, 395, 355], "label": "Rice Leaf Roller", "conf": 0.91, "color": "#10b981"}
        ]
    },
    {
        "id": "micro_aphids",
        "title": "Case 3: Micro-scale Pest Localization",
        "pest_name": "Aphids (Aphis gossypii)",
        "benchmark": "IP102",
        "challenge": "Extremely small targets (<32x32 px) on crop leaves.",
        "description": "Baseline misses the third micro-pest due to noisy full-channel attention maps; CR-HSDPA cleanly captures all 3 micro-scale targets.",
        "baseline_boxes": [
            {"box": [210, 160, 270, 220], "label": "Aphids", "conf": 0.71, "color": "#f59e0b"},
            {"box": [320, 270, 380, 330], "label": "Aphids", "conf": 0.68, "color": "#f59e0b"}
        ],
        "cr_boxes": [
            {"box": [210, 160, 268, 218], "label": "Aphids", "conf": 0.88, "color": "#10b981"},
            {"box": [322, 272, 378, 328], "label": "Aphids", "conf": 0.85, "color": "#10b981"},
            {"box": [410, 360, 465, 415], "label": "Aphids", "conf": 0.82, "color": "#10b981"}
        ]
    },
    {
        "id": "multiclass_field",
        "title": "Case 4: Multi-Class Field Infestation",
        "pest_name": "Asiatic Corn Borer & Plant Bug",
        "benchmark": "IP102",
        "challenge": "Distinct pest species co-occurring in outdoor field crops.",
        "description": "Both models detect targets, but CR-HSDPA provides tighter bounding boxes and elevated classification confidence.",
        "baseline_boxes": [
            {"box": [140, 180, 290, 340], "label": "Asiatic Corn Borer", "conf": 0.79, "color": "#f59e0b"},
            {"box": [360, 260, 490, 410], "label": "Plant Bug", "conf": 0.74, "color": "#f59e0b"}
        ],
        "cr_boxes": [
            {"box": [145, 185, 285, 335], "label": "Asiatic Corn Borer", "conf": 0.93, "color": "#10b981"},
            {"box": [365, 265, 485, 405], "label": "Plant Bug", "conf": 0.90, "color": "#10b981"}
        ]
    }
]
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_config.py`
Expected: PASS

- [ ] **Step 5: Commit**
Run: `git add config.py tests/test_config.py; git commit -m "feat: add benchmark data and configuration module"`

---

### Task 2: Preset Agricultural Sample Image Generator

**Files:**
- Create: `d:/Thesisss/Thesis_Mockup/sample_generator.py`
- Test: `d:/Thesisss/Thesis_Mockup/tests/test_samples.py`

**Interfaces:**
- Produces:
  - `generate_preset_images(output_dir="assets/presets") -> dict`: Generates high-quality, realistic visual representations of the 4 pest scenarios at 640×640 resolution with foliage, plant structures, and insects.
  - Returns dictionary of `{case_id: file_path}`.

- [ ] **Step 1: Write the failing test**
Create `tests/test_samples.py`:
```python
import os
import pytest
from PIL import Image
from sample_generator import generate_preset_images

def test_preset_image_generation(tmp_path):
    output_dir = str(tmp_path / "presets")
    image_paths = generate_preset_images(output_dir)
    
    assert len(image_paths) == 4
    for case_id, path in image_paths.items():
        assert os.path.exists(path)
        with Image.open(path) as img:
            assert img.size == (640, 640)
            assert img.format in ["JPEG", "PNG"]
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_samples.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'sample_generator'`

- [ ] **Step 3: Implement `sample_generator.py`**
Create `sample_generator.py` using Pillow with realistic plant textures, leaf veins, and insect morphology for all 4 cases.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_samples.py`
Expected: PASS

- [ ] **Step 5: Commit**
Run: `git add sample_generator.py tests/test_samples.py; git commit -m "feat: add agricultural pest sample image generator"`

---

### Task 3: Bounding Box Rendering & Detection Simulation Engine

**Files:**
- Create: `d:/Thesisss/Thesis_Mockup/engine.py`
- Test: `d:/Thesisss/Thesis_Mockup/tests/test_engine.py`

**Interfaces:**
- Produces:
  - `draw_yolo_detections(image: Image.Image, boxes: list[dict], conf_threshold: float, show_labels: bool, show_conf: bool, box_width: int) -> Image.Image`
  - `simulate_inference_latency(model_type: str, cr_ratio: float) -> tuple[float, float]` (returns latency ms and FPS)
  - `process_custom_image(image: Image.Image, benchmark: str, conf_threshold: float, cr_ratio: float) -> tuple[list[dict], list[dict]]` (returns baseline_boxes, cr_boxes for uploaded photos)

- [ ] **Step 1: Write the failing test**
Create `tests/test_engine.py`:
```python
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
    
    assert 27.0 <= base_lat <= 30.0
    assert 17.5 <= cr_lat <= 20.0
    assert cr_fps > base_fps

def test_process_custom_image():
    img = Image.new("RGB", (800, 600), color=(80, 120, 80))
    base_boxes, cr_boxes = process_custom_image(img, benchmark="IP102", conf_threshold=0.30, cr_ratio=0.50)
    assert len(base_boxes) >= 1
    assert len(cr_boxes) >= 1
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_engine.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine'`

- [ ] **Step 3: Implement `engine.py`**
Implement PIL-based drawing with anti-aliased rectangles, rounded pill headers for labels and confidence percentages, color coding, and latency jitter simulation.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_engine.py`
Expected: PASS

- [ ] **Step 5: Commit**
Run: `git add engine.py tests/test_engine.py; git commit -m "feat: add yolo bbox rendering and detection engine"`

---

### Task 4: Interactive Architecture Explainer & Ablation Charts

**Files:**
- Create: `d:/Thesisss/Thesis_Mockup/components.py`
- Test: `d:/Thesisss/Thesis_Mockup/tests/test_components.py`

**Interfaces:**
- Produces:
  - `render_metric_card(label: str, baseline_val: str, cr_val: str, delta: str, is_positive: bool)`
  - `plot_ablation_pareto_curve() -> matplotlib.figure.Figure`
  - `render_architecture_diagram_html() -> str`

- [ ] **Step 1: Write the failing test**
Create `tests/test_components.py`:
```python
import pytest
import matplotlib.figure
from components import plot_ablation_pareto_curve, render_architecture_diagram_html

def test_plot_ablation_pareto_curve():
    fig = plot_ablation_pareto_curve()
    assert isinstance(fig, matplotlib.figure.Figure)

def test_render_architecture_diagram_html():
    html = render_architecture_diagram_html()
    assert "HSDPA" in html
    assert "CR-HSDPA" in html
    assert "1×1 Conv" in html
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_components.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'components'`

- [ ] **Step 3: Implement `components.py`**
Implement the Pareto curve visualization (mAP@50 vs GFLOPs), architectural HTML flow diagram comparing baseline full-channel HSDPA vs. CR-HSDPA bottleneck, and helper card renderers.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_components.py`
Expected: PASS

- [ ] **Step 5: Commit**
Run: `git add components.py tests/test_components.py; git commit -m "feat: add architecture visualization and ablation components"`

---

### Task 5: Streamlit Application Assembly & Verification

**Files:**
- Create: `d:/Thesisss/Thesis_Mockup/app.py`
- Create: `d:/Thesisss/Thesis_Mockup/requirements.txt`

- [ ] **Step 1: Create `requirements.txt`**
List: `streamlit>=1.30.0`, `pillow>=10.0.0`, `numpy>=1.24.0`, `matplotlib>=3.8.0`, `pytest>=7.0.0`.

- [ ] **Step 2: Implement `app.py`**
Build the complete Streamlit UI integrating:
1. Academic Header & Metadata Bar.
2. Sidebar controls (Benchmark switch, Preset gallery vs Custom upload, Confidence & NMS sliders, CR ratio).
3. Side-by-side visual detection columns (Baseline vs Proposed).
4. Interactive KPI Metric comparison grid with delta badges.
5. Interactive Ablation Study tab (table & Pareto chart).
6. "Why Our Model Wins" Architectural breakdown tab.
7. Raw Detections Inspection table.

- [ ] **Step 3: Verify execution and test suite**
Run:
`pytest tests/ -v`
`streamlit run app.py --server.headless=true`
Verify that all unit tests pass and app launches without exceptions.

- [ ] **Step 4: Commit and tag**
Run: `git add app.py requirements.txt; git commit -m "feat: complete thesis pestnet mockup streamlit dashboard"`
