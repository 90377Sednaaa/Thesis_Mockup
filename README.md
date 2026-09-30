# Modified Attention-PestNet (CR-HSDPA) — Insect Pest Detection Prototype

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive thesis prototype and model evaluation dashboard for **Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention (CR-HSDPA)** for efficient agricultural insect pest detection.

---

## 📌 Research Overview

* **Thesis Title:** *A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention for Efficient Insect Pest Detection*
* **Authors:** Lean Adrian Murillo, James Oliver C. Mendoza, DM Rashid P. Ferrer, Charisse P. Barbosa
* **Institution:** Department of Computer Science, University of Mindanao, Matina, Davao City, Philippines

### The Core Problem & Our Innovation
In the baseline **Attention-PestNet** (Doan et al., 2026), 4-level Top-Down Attention (TDA) is executed on full channel dimensions ($C$) across three repeated neck blocks. This creates substantial computational overhead and feature redundancy.

Our proposed **CR-HSDPA** module introduces:
1. **Pre-Attention 1×1 Bottleneck:** Linearly compresses incoming channels from $C \to C_r$ ($r = 0.50$).
2. **Compressed 4-Level TDA:** Attention computes on half the channels, slashing quadratic complexity.
3. **Channel Restoration:** A post-attention 1×1 convolution seamlessly restores the channel dimension back to $C$.

---

## 📊 Benchmark Results

Evaluated on the **IP102** (102 classes) and **R2000** (16 rice pests) benchmarks on an **NVIDIA T4 GPU** (FP32, Batch Size 1, 640×640 input resolution):

| Evaluation Metric | Baseline Attention-PestNet | Proposed CR-HSDPA ($r=0.50$) | Difference / Impact |
| :--- | :---: | :---: | :---: |
| **mAP@50 (%)** | 68.67% | **69.40%** | **+0.73%** (Higher accuracy) |
| **mAP@50:95 (%)** | 44.17% | **44.80%** | **+0.63%** (Tighter boundary overlap) |
| **Complexity (GFLOPs)** | 162.70 | **118.40** | **-27.2%** (Huge computational savings) |
| **Parameters (M)** | 79.66 M | **58.20 M** | **-26.9%** (-21.46M lighter) |
| **Forward Latency** | 28.4 ms | **18.6 ms** | **-34.5%** (Substantially faster) |
| **Throughput (FPS)** | 35.2 FPS | **53.8 FPS** | **+52.8%** (Smooth real-time) |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/90377Sednaaa/Thesis_Mockup.git
cd Thesis_Mockup
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Unit Tests
```bash
pytest tests/ -v
```

---

## 📁 Repository Structure

```
Thesis_Mockup/
├── app.py                     # Streamlit application dashboard
├── config.py                  # Benchmark metrics & preset configurations
├── engine.py                  # YOLO-style bbox drawing & latency simulation
├── components.py              # Pareto curve plot & architecture diagram
├── sample_generator.py        # Preset image loader & manager
├── requirements.txt           # Python package dependencies
├── assets/presets/            # Real macro agricultural pest photography
├── tests/                     # Pytest unit test suite
└── docs/                      # Design specs and implementation plans
```

---

## 📄 Citation & Attribution

If you use or reference this work in your research, please cite:
```bibtex
@article{murillo2026modifiedpestnet,
  title={A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention for Efficient Insect Pest Detection},
  author={Murillo, Lean Adrian and Mendoza, James Oliver C. and Ferrer, DM Rashid P. and Barbosa, Charisse P.},
  institution={University of Mindanao},
  year={2026}
}
```
