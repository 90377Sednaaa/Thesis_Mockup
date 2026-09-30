"""
Configuration and benchmark metrics for the thesis:
'A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention for Efficient Insect Pest Detection'

Authors: Lean Adrian Murillo, James Oliver C. Mendoza, DM Rashid P. Ferrer, Charisse P. Barbosa
Institution: University of Mindanao, Davao City, Philippines
"""

BENCHMARKS = {
    "IP102": {
        "title": "IP102 (General Insect Pest Benchmark - 102 Classes)",
        "num_classes": 102,
        "input_size": "640x640",
        "description": "Large-scale benchmark covering 75,222 images across 102 insect pest classes.",
        "baseline": {
            "name": "Attention-PestNet (Baseline)",
            "architecture": "Standard HSDPA (Full Channels C)",
            "map50": 68.67,
            "map50_95": 44.17,
            "params_m": 79.66,
            "gflops": 162.70,
            "model_size_mb": 152.3,
            "latency_ms": 28.4,
            "fps": 35.2,
        },
        "cr_hsdpa": {
            "name": "Our Model (CR-HSDPA)",
            "architecture": "CR-HSDPA (Channel-Reduced Bottleneck)",
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
        "description": "Rice crop benchmark containing 2,046 images across 16 major rice insect pests.",
        "baseline": {
            "name": "Attention-PestNet (Baseline)",
            "architecture": "Standard HSDPA (Full Channels C)",
            "map50": 82.60,
            "map50_95": 68.20,
            "params_m": 79.66,
            "gflops": 162.70,
            "model_size_mb": 152.3,
            "latency_ms": 28.1,
            "fps": 35.6,
        },
        "cr_hsdpa": {
            "name": "Our Model (CR-HSDPA)",
            "architecture": "CR-HSDPA (Channel-Reduced Bottleneck)",
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
        "status": "Architecture Control"
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
        "status": "Our Model (Best Balance)"
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
        "status": "Aggressive Compression"
    }
]

PRESET_CASES = [
    {
        "id": "clustered_bph",
        "title": "Case 1: Crowded Pests (Brown Planthopper)",
        "pest_name": "Brown Planthopper",
        "benchmark": "R2000",
        "filename": "clustered_bph.jpg",
        "challenge": "Multiple overlapping insects clustered tightly along rice stems.",
        "description": "Baseline struggles with crowded insects and predicts lower confidence (64%–75%). Our model cleanly detects each insect with higher confidence (86%–93%).",
        "baseline_boxes": [
            {"box": [255, 145, 340, 235], "label": "Brown Planthopper", "conf": 0.72, "color": "#f59e0b"},
            {"box": [285, 215, 365, 315], "label": "Brown Planthopper", "conf": 0.68, "color": "#f59e0b"},
            {"box": [230, 320, 310, 415], "label": "Brown Planthopper", "conf": 0.75, "color": "#f59e0b"},
            {"box": [260, 385, 340, 495], "label": "Brown Planthopper", "conf": 0.64, "color": "#f59e0b"}
        ],
        "cr_boxes": [
            {"box": [260, 150, 335, 230], "label": "Brown Planthopper", "conf": 0.90, "color": "#10b981"},
            {"box": [290, 220, 360, 310], "label": "Brown Planthopper", "conf": 0.88, "color": "#10b981"},
            {"box": [235, 325, 305, 410], "label": "Brown Planthopper", "conf": 0.93, "color": "#10b981"},
            {"box": [265, 390, 335, 490], "label": "Brown Planthopper", "conf": 0.86, "color": "#10b981"}
        ]
    },
    {
        "id": "camouflaged_leafroller",
        "title": "Case 2: Camouflaged Pest (Rice Leaf Roller)",
        "pest_name": "Rice Leaf Roller",
        "benchmark": "R2000",
        "filename": "camouflaged_leafroller.jpg",
        "challenge": "Pest blends inside a curled green rice leaf blade.",
        "description": "Baseline gets confused by the folded leaf edge and predicts a False Positive (52% conf). Our model ignores the leaf fold and only detects the actual pest (94% conf).",
        "baseline_boxes": [
            {"box": [230, 250, 435, 430], "label": "Rice Leaf Roller", "conf": 0.76, "color": "#f59e0b"},
            {"box": [460, 140, 560, 260], "label": "Rice Leaf Roller [False Alarm]", "conf": 0.52, "color": "#ef4444"}
        ],
        "cr_boxes": [
            {"box": [240, 260, 425, 420], "label": "Rice Leaf Roller", "conf": 0.94, "color": "#10b981"}
        ]
    },
    {
        "id": "micro_aphids",
        "title": "Case 3: Small Pests (Aphids)",
        "pest_name": "Aphids",
        "benchmark": "IP102",
        "filename": "micro_aphids.jpg",
        "challenge": "Very small pests across the leaf surface.",
        "description": "Baseline misses the smaller aphids due to background leaf noise. Our model captures all of them with strong confidence.",
        "baseline_boxes": [
            {"box": [280, 235, 350, 305], "label": "Aphids", "conf": 0.70, "color": "#f59e0b"},
            {"box": [315, 295, 380, 370], "label": "Aphids", "conf": 0.67, "color": "#f59e0b"}
        ],
        "cr_boxes": [
            {"box": [285, 240, 345, 300], "label": "Aphids", "conf": 0.89, "color": "#10b981"},
            {"box": [320, 300, 375, 365], "label": "Aphids", "conf": 0.87, "color": "#10b981"},
            {"box": [310, 370, 368, 440], "label": "Aphids", "conf": 0.84, "color": "#10b981"},
            {"box": [255, 440, 305, 495], "label": "Aphids", "conf": 0.81, "color": "#10b981"}
        ]
    },
    {
        "id": "multiclass_field",
        "title": "Case 4: Field Crop Pest (Armyworm Caterpillar)",
        "pest_name": "Armyworm",
        "benchmark": "IP102",
        "filename": "multiclass_field.jpg",
        "challenge": "Large foliage pest on green crop leaf.",
        "description": "Baseline detects the pest with a loose box and 77% confidence. Our model fits a tighter box with 95% confidence.",
        "baseline_boxes": [
            {"box": [120, 120, 560, 550], "label": "Armyworm", "conf": 0.77, "color": "#f59e0b"}
        ],
        "cr_boxes": [
            {"box": [135, 135, 545, 535], "label": "Armyworm", "conf": 0.95, "color": "#10b981"}
        ]
    }
]

CLASS_COLORS = {
    "Brown Planthopper": "#3b82f6",
    "Rice Leaf Roller": "#8b5cf6",
    "Aphids": "#ec4899",
    "Armyworm": "#10b981",
    "Asiatic Corn Borer": "#f97316"
}
