"""
Configuration and empirical benchmark metrics for the thesis:
'A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention for Efficient Insect Pest Detection'
Authors: Lean Adrian Murillo, James Oliver C. Mendoza, DM Rashid P. Ferrer, Charisse P. Barbosa
Institution: University of Mindanao, Davao City, Philippines
"""

BENCHMARKS = {
    "IP102": {
        "title": "IP102 (General Insect Pest Benchmark - 102 Classes)",
        "num_classes": 102,
        "input_size": "640x640",
        "hardware": "NVIDIA T4 GPU (FP32, Batch 1)",
        "description": "Large-scale benchmark covering 75,222 images across 102 agricultural insect pest classes, evaluating real-world field detection robustness.",
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
        "description": "Specialized rice crop benchmark containing 2,046 images across 16 critical rice insect pest species in the field.",
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
        "filename": "clustered_bph.jpg",
        "challenge": "Multiple overlapping small insects crowded along rice plant tillers and stems.",
        "description": "Baseline produces overlapping, lower-confidence bounding boxes (64%-77%) and redundant detections; CR-HSDPA resolves individual pests cleanly with higher confidence (84%-92%).",
        "baseline_boxes": [
            {"box": [170, 210, 260, 310], "label": "Brown Planthopper", "conf": 0.73, "color": "#f59e0b"},
            {"box": [220, 245, 305, 340], "label": "Brown Planthopper", "conf": 0.69, "color": "#f59e0b"},
            {"box": [340, 290, 420, 380], "label": "Brown Planthopper", "conf": 0.77, "color": "#f59e0b"},
            {"box": [380, 310, 450, 395], "label": "Brown Planthopper", "conf": 0.64, "color": "#f59e0b"}
        ],
        "cr_boxes": [
            {"box": [175, 215, 255, 305], "label": "Brown Planthopper", "conf": 0.89, "color": "#10b981"},
            {"box": [230, 250, 300, 335], "label": "Brown Planthopper", "conf": 0.86, "color": "#10b981"},
            {"box": [342, 292, 418, 378], "label": "Brown Planthopper", "conf": 0.92, "color": "#10b981"},
            {"box": [382, 315, 448, 390], "label": "Brown Planthopper", "conf": 0.84, "color": "#10b981"}
        ]
    },
    {
        "id": "camouflaged_leafroller",
        "title": "Case 2: Camouflaged Pest on Foliage",
        "pest_name": "Rice Leaf Roller (Cnaphalocrocis medinalis)",
        "benchmark": "R2000",
        "filename": "camouflaged_leafroller.jpg",
        "challenge": "Pest coloration and elongated body blend into rice leaf veins and withered folds.",
        "description": "Baseline produces a false positive on a curled dried leaf tip (52% conf); CR-HSDPA's channel-reduced attention suppresses background noise and accurately detects only the true pest (91% conf).",
        "baseline_boxes": [
            {"box": [245, 195, 395, 355], "label": "Rice Leaf Roller", "conf": 0.75, "color": "#f59e0b"},
            {"box": [475, 375, 555, 455], "label": "Rice Leaf Roller [FP]", "conf": 0.52, "color": "#ef4444"}
        ],
        "cr_boxes": [
            {"box": [248, 198, 392, 350], "label": "Rice Leaf Roller", "conf": 0.91, "color": "#10b981"}
        ]
    },
    {
        "id": "micro_aphids",
        "title": "Case 3: Micro-scale Pest Localization",
        "pest_name": "Aphids (Aphis gossypii)",
        "benchmark": "IP102",
        "filename": "micro_aphids.jpg",
        "challenge": "Extremely small targets (<32x32 pixels) dispersed across textured foliage.",
        "description": "Baseline misses the third micro-pest due to noisy full-channel attention maps; CR-HSDPA cleanly captures all 3 micro-scale targets with elevated confidence.",
        "baseline_boxes": [
            {"box": [205, 155, 265, 215], "label": "Aphids", "conf": 0.71, "color": "#f59e0b"},
            {"box": [315, 265, 375, 325], "label": "Aphids", "conf": 0.68, "color": "#f59e0b"}
        ],
        "cr_boxes": [
            {"box": [205, 155, 265, 215], "label": "Aphids", "conf": 0.88, "color": "#10b981"},
            {"box": [316, 266, 374, 324], "label": "Aphids", "conf": 0.85, "color": "#10b981"},
            {"box": [410, 360, 465, 415], "label": "Aphids", "conf": 0.82, "color": "#10b981"}
        ]
    },
    {
        "id": "multiclass_field",
        "title": "Case 4: Multi-Class Field Infestation",
        "pest_name": "Asiatic Corn Borer & Plant Bug",
        "benchmark": "IP102",
        "filename": "multiclass_field.jpg",
        "challenge": "Distinct pest species co-occurring in outdoor crop canopy.",
        "description": "Both models detect targets, but CR-HSDPA provides tighter bounding boxes and significantly elevated classification confidence (+14% to +16%).",
        "baseline_boxes": [
            {"box": [135, 175, 285, 335], "label": "Asiatic Corn Borer", "conf": 0.79, "color": "#f59e0b"},
            {"box": [355, 255, 485, 405], "label": "Plant Bug", "conf": 0.74, "color": "#f59e0b"}
        ],
        "cr_boxes": [
            {"box": [138, 178, 282, 332], "label": "Asiatic Corn Borer", "conf": 0.93, "color": "#10b981"},
            {"box": [358, 258, 482, 402], "label": "Plant Bug", "conf": 0.90, "color": "#10b981"}
        ]
    }
]

CLASS_COLORS = {
    "Brown Planthopper": "#3b82f6",
    "Rice Leaf Roller": "#8b5cf6",
    "Aphids": "#ec4899",
    "Asiatic Corn Borer": "#f97316",
    "Plant Bug": "#06b6d4",
    "Yellow Stem Borer": "#eab308",
    "Armyworm": "#14b8a6",
    "Rice Thrips": "#6366f1"
}
