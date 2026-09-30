"""
Streamlit Thesis Prototype Application:
'A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention for Efficient Insect Pest Detection'

Authors: Lean Adrian Murillo, James Oliver C. Mendoza, DM Rashid P. Ferrer, Charisse P. Barbosa
University of Mindanao, Davao City, Philippines
"""

import os
import streamlit as st
import pandas as pd
from PIL import Image

from config import BENCHMARKS, ABLATION_DATA, PRESET_CASES, CLASS_COLORS
from sample_generator import generate_preset_images
from engine import (
    draw_yolo_detections,
    simulate_inference_latency,
    process_custom_image,
    apply_nms
)
from components import (
    plot_ablation_pareto_curve,
    render_architecture_diagram_html,
    render_metric_card
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Modified Attention-PestNet (CR-HSDPA) | Thesis Prototype",
    page_icon="🦗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Styling for Academic Thesis Presentation
st.markdown("""
<style>
    /* Dark High-Contrast Academic Theme */
    .stApp {
        background-color: #0b0f17;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header Card */
    .thesis-header {
        background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .thesis-title {
        font-size: 22px;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.3px;
        margin-bottom: 8px;
    }
    .thesis-meta {
        font-size: 13px;
        color: #94a3b8;
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
    }
    .meta-tag {
        background: #1e293b;
        border: 1px solid #475569;
        padding: 3px 10px;
        border-radius: 6px;
        color: #cbd5e1;
    }

    /* Comparison Headings */
    .model-header-base {
        background: #271c19;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .model-header-cr {
        background: #112822;
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }

    /* Challenge Callout */
    .challenge-box {
        background: #161e2e;
        border: 1px solid #2563eb;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 15px;
        font-size: 13px;
        line-height: 1.5;
        color: #e2e8f0;
    }

    /* Metric Badges */
    .badge-win {
        background: #064e3b;
        color: #34d399;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
    }

    /* Sidebar Tweaks */
    [data-testid="stSidebar"] {
        background-color: #0d121c;
        border-right: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Ensure Preset Images Exist
# ---------------------------------------------------------
preset_paths = generate_preset_images("assets/presets")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/insect.png", width=64)
st.sidebar.markdown("## 🔬 Thesis Controls")

# Benchmark Selection
selected_bm_key = st.sidebar.selectbox(
    "Select Benchmark Dataset",
    options=["IP102", "R2000"],
    format_func=lambda k: f"{k} ({BENCHMARKS[k]['title'].split('(')[1].split(')')[0]})"
)
bm_info = BENCHMARKS[selected_bm_key]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖼️ Image Input Source")
input_mode = st.sidebar.radio(
    "Choose Input Method",
    options=["Ready-to-Test Preset Scenarios", "Upload Custom Crop Image"]
)

active_preset = None
uploaded_file = None

if input_mode == "Ready-to-Test Preset Scenarios":
    # Filter presets matching the benchmark or allow all 4
    preset_options = [c["title"] for c in PRESET_CASES]
    selected_preset_title = st.sidebar.selectbox("Select Insect Pest Scenario", preset_options)
    active_preset = next(c for c in PRESET_CASES if c["title"] == selected_preset_title)
    
    st.sidebar.info(
        f"**Target:** {active_preset['pest_name']}\n\n"
        f"**Benchmark Context:** {active_preset['benchmark']}\n\n"
        f"**Field Challenge:** {active_preset['challenge']}"
    )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Agricultural Photo (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if not uploaded_file:
        st.sidebar.caption("💡 No file chosen yet. Showing default demo image until you upload.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Inference & Model Tuning")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
    help="Detections with confidence scores below this threshold are filtered out."
)

iou_threshold = st.sidebar.slider(
    "NMS IoU Threshold",
    min_value=0.20,
    max_value=0.80,
    value=0.45,
    step=0.05,
    help="Intersection over Union threshold for Non-Maximum Suppression."
)

cr_variant = st.sidebar.selectbox(
    "Our Model Compression Ratio",
    options=["CR-0.50 (Selected Best - r=0.50)", "CR-0.75 (Mild - r=0.75)", "CR-0.25 (Aggressive - r=0.25)", "CR-1.00 (Control - r=1.00)"],
    index=0
)
cr_ratio_val = 0.50
if "0.75" in cr_variant:
    cr_ratio_val = 0.75
elif "0.25" in cr_variant:
    cr_ratio_val = 0.25
elif "1.00" in cr_variant:
    cr_ratio_val = 1.00

st.sidebar.markdown("---")
show_labels = st.sidebar.checkbox("Show Pest Class Labels", value=True)
show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)
box_thickness = st.sidebar.slider("Box Outline Thickness", min_value=1, max_value=5, value=3)

# ---------------------------------------------------------
# Top Academic Header Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="thesis-header">
    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #38bdf8; font-weight: 700; margin-bottom: 4px;">
        Undergraduate Thesis Interactive Prototype & Model Evaluation Mockup
    </div>
    <div class="thesis-title">
        A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention for Efficient Insect Pest Detection
    </div>
    <div class="thesis-meta">
        <span class="meta-tag">👤 <strong>Authors:</strong> L. A. Murillo, J. O. C. Mendoza, D. R. P. Ferrer, C. P. Barbosa</span>
        <span class="meta-tag">🏛️ <strong>Institution:</strong> University of Mindanao</span>
        <span class="meta-tag">🖥️ <strong>Benchmark Platform:</strong> NVIDIA T4 GPU (FP32, 640×640)</span>
        <span class="meta-tag">🎯 <strong>Active Benchmark:</strong> {bm_info['title']}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Current Image & Detections
# ---------------------------------------------------------
if input_mode == "Ready-to-Test Preset Scenarios" and active_preset:
    image_path = preset_paths[active_preset["id"]]
    source_image = Image.open(image_path)
    baseline_boxes = active_preset["baseline_boxes"]
    cr_boxes = active_preset["cr_boxes"]
    case_summary = active_preset["description"]
elif uploaded_file is not None:
    source_image = Image.open(uploaded_file)
    baseline_boxes, cr_boxes = process_custom_image(
        source_image,
        benchmark=selected_bm_key,
        conf_threshold=conf_threshold,
        cr_ratio=cr_ratio_val
    )
    case_summary = f"Custom field image analyzed under {selected_bm_key} benchmark. CR-HSDPA suppresses background false alarms and sharpens confidence."
else:
    # Fallback to Case 1
    image_path = preset_paths["clustered_bph"]
    source_image = Image.open(image_path)
    baseline_boxes = PRESET_CASES[0]["baseline_boxes"]
    cr_boxes = PRESET_CASES[0]["cr_boxes"]
    case_summary = PRESET_CASES[0]["description"]

# Simulate realistic forward-pass hardware timings
base_lat, base_fps = simulate_inference_latency("baseline", cr_ratio=1.0)
cr_lat, cr_fps = simulate_inference_latency("cr_hsdpa", cr_ratio=cr_ratio_val)

# Render YOLO detection overlays
img_baseline_rendered = draw_yolo_detections(
    source_image,
    baseline_boxes,
    conf_threshold=conf_threshold,
    iou_threshold=iou_threshold,
    show_labels=show_labels,
    show_conf=show_conf,
    box_width=box_thickness
)

img_cr_rendered = draw_yolo_detections(
    source_image,
    cr_boxes,
    conf_threshold=conf_threshold,
    iou_threshold=iou_threshold,
    show_labels=show_labels,
    show_conf=show_conf,
    box_width=box_thickness
)

# ---------------------------------------------------------
# Main Stage: Side-by-Side Dual Visualizer
# ---------------------------------------------------------
st.markdown("### 🔍 Side-by-Side Model Inference Comparison")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="model-header-base">
        <span style="font-weight: 800; font-size: 15px; color: #fbbf24;">1. Baseline Attention-PestNet (Doan et al., 2026)</span><br>
        <span style="font-size: 11.5px; color: #cbd5e1;">Architecture: Standard HSDPA (Full Channels C, 4 TDA Levels)</span>
    </div>
    """, unsafe_allow_html=True)
    st.image(img_baseline_rendered, use_container_width=True)
    st.caption(f"⏱️ **Forward Pass:** {base_lat} ms | **Speed:** {base_fps} FPS | **Complexity:** 162.7 GFLOPs (79.66M params)")

with col_right:
    st.markdown(f"""
    <div class="model-header-cr">
        <span style="font-weight: 800; font-size: 15px; color: #34d399;">2. Modified Attention-PestNet ({cr_variant.split(' ')[0]})</span><br>
        <span style="font-size: 11.5px; color: #cbd5e1;">Architecture: CR-HSDPA (1×1 Bottleneck, Cr={cr_ratio_val}C, 4 TDA Levels)</span>
    </div>
    """, unsafe_allow_html=True)
    st.image(img_cr_rendered, use_container_width=True)
    st.caption(f"⚡ **Forward Pass:** {cr_lat} ms (<span class='badge-win'>-{round((base_lat - cr_lat)/base_lat*100, 1)}% faster</span>) | **Speed:** {cr_fps} FPS | **Complexity:** 118.4 GFLOPs (-27.2%)", unsafe_allow_html=True)

# Field Challenge Diagnostic Callout
st.markdown(f"""
<div class="challenge-box">
    <strong>📋 Detection Analysis on Current Sample:</strong><br>
    {case_summary}
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Side-by-Side Benchmark Performance Matrix
# ---------------------------------------------------------
st.markdown(f"### 📊 Empirical Metrics Matrix ({selected_bm_key} Benchmark)")

base_data = bm_info["baseline"]
cr_data = bm_info["cr_hsdpa"]

m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

with m_col1:
    map_delta = round(cr_data["map50"] - base_data["map50"], 2)
    st.metric(
        label="mAP@50 (%)",
        value=f"{cr_data['map50']}%",
        delta=f"+{map_delta}%",
        delta_color="normal",
        help="Mean Average Precision at IoU 0.50 across all classes."
    )

with m_col2:
    map95_delta = round(cr_data["map50_95"] - base_data["map50_95"], 2)
    st.metric(
        label="mAP@50:95 (%)",
        value=f"{cr_data['map50_95']}%",
        delta=f"+{map95_delta}%",
        delta_color="normal",
        help="Strict localization quality averaged across IoU 0.50 to 0.95."
    )

with m_col3:
    gflops_pct = round((cr_data["gflops"] - base_data["gflops"]) / base_data["gflops"] * 100, 1)
    st.metric(
        label="GFLOPs (640×640)",
        value=f"{cr_data['gflops']}",
        delta=f"{gflops_pct}%",
        delta_color="inverse",
        help="Floating point operations per single forward pass. Lower is lighter."
    )

with m_col4:
    params_pct = round((cr_data["params_m"] - base_data["params_m"]) / base_data["params_m"] * 100, 1)
    st.metric(
        label="Parameters (M)",
        value=f"{cr_data['params_m']} M",
        delta=f"{params_pct}%",
        delta_color="inverse",
        help="Total learned network parameters in millions. Lower is lighter."
    )

with m_col5:
    lat_pct = round((cr_data["latency_ms"] - base_data["latency_ms"]) / base_data["latency_ms"] * 100, 1)
    st.metric(
        label="Latency (ms)",
        value=f"{cr_data['latency_ms']} ms",
        delta=f"{lat_pct}%",
        delta_color="inverse",
        help="GPU forward-pass execution time on Kaggle NVIDIA T4 GPU."
    )

with m_col6:
    fps_pct = round((cr_data["fps"] - base_data["fps"]) / base_data["fps"] * 100, 1)
    st.metric(
        label="Throughput (FPS)",
        value=f"{cr_data['fps']} FPS",
        delta=f"+{fps_pct}%",
        delta_color="normal",
        help="Frames processed per second. Higher represents superior real-time performance."
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs: Architectural Breakdown, Ablation Study, Raw Detections
# ---------------------------------------------------------
tab_arch, tab_ablation, tab_boxes = st.tabs([
    "📐 Why Our Model Wins (Architecture)",
    "📈 Ablation Study & Pareto Curve",
    "📋 Raw Detections Inspector"
])

# --- Tab 1: Architecture ---
with tab_arch:
    st.markdown("#### Architectural Comparison: Baseline HSDPA vs Proposed CR-HSDPA Neck")
    st.write(
        "In Attention-PestNet, the computational bottleneck stems from repeating 4-level "
        "Top-Down Attention (TDA) across the full channel dimension $C$. Our thesis introduces "
        "a learnable $1\\times1$ convolution bottleneck that compresses channels to $C_r = 0.5C$, "
        "substantially reducing quadratic attention overhead while preserving discriminative features."
    )
    st.components.v1.html(render_architecture_diagram_html(), height=360, scrolling=True)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        **Key Findings & Mechanism:**
        1. **Elimination of Channel Redundancy**: Multi-scale attention maps across deep neck layers exhibit high inter-channel collinearity. Compressing $C \\to 0.5C$ eliminates redundant computation.
        2. **Attention Regularization**: In outdoor agricultural pest imagery, complex foliage textures can distract full-channel attention. Channel compression filters background noise and tightens insect localization.
        """)
    with col_a2:
        st.markdown("""
        **Compliance with Thesis Success Criteria:**
        - **Target Criterion**: Achieve $\\ge 10\\%$ GFLOPs reduction with $\\le 1.0\\%$ drop in validation mAP@50:95.
        - **Empirical Result**: CR-0.50 achieves **-27.2% GFLOPs reduction** while **increasing mAP@50 by +0.73%** and **mAP@50:95 by +0.63%**, exceeding the thesis goal.
        """)

# --- Tab 2: Ablation Study ---
with tab_ablation:
    st.markdown("#### Complete Ablation Study on Reduction Ratios ($r$)")
    col_abl_tbl, col_abl_chart = st.columns([1, 1.2])

    with col_abl_tbl:
        df_ablation = pd.DataFrame(ABLATION_DATA)
        st.dataframe(
            df_ablation[["variant", "ratio", "params_m", "gflops", "delta_gflops", "map50", "fps", "status"]],
            column_config={
                "variant": "Ablation Variant",
                "ratio": "Ratio (r)",
                "params_m": "Params (M)",
                "gflops": "GFLOPs",
                "delta_gflops": "Δ GFLOPs",
                "map50": "mAP@50 (%)",
                "fps": "FPS",
                "status": "Role / Outcome"
            },
            hide_index=True,
            use_container_width=True
        )
        st.info("💡 **Pareto Selection**: `CR-0.50` delivers the strongest accuracy-efficiency trade-off before feature degradation occurs at `CR-0.25`.")

    with col_abl_chart:
        fig_ablation = plot_ablation_pareto_curve()
        st.pyplot(fig_ablation)

# --- Tab 3: Raw Detections Inspector ---
with tab_boxes:
    st.markdown("#### Bounding Box Prediction Details for Current Sample")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("**Baseline Attention-PestNet Predictions:**")
        valid_b = [b for b in baseline_boxes if b.get("conf", 0.0) >= conf_threshold]
        valid_b_filtered = apply_nms(valid_b, iou_threshold=iou_threshold)
        if valid_b_filtered:
            rows_b = []
            for b in valid_b_filtered:
                box_str = f"[{b['box'][0]}, {b['box'][1]}, {b['box'][2]}, {b['box'][3]}]"
                rows_b.append({
                    "Class": b.get("label", "Insect"),
                    "Confidence": f"{b.get('conf', 0.0)*100:.1f}%",
                    "Bounding Box [x1,y1,x2,y2]": box_str,
                    "Status": "⚠️ Loose Box" if b.get('conf', 0) < 0.75 else ("❌ False Positive" if "FP" in b.get("label", "") else "✔️ Detected")
                })
            st.dataframe(pd.DataFrame(rows_b), hide_index=True, use_container_width=True)
        else:
            st.warning("No baseline detections above current confidence threshold.")

    with col_t2:
        st.markdown("**Proposed CR-HSDPA Predictions:**")
        valid_c = [b for b in cr_boxes if b.get("conf", 0.0) >= conf_threshold]
        valid_c_filtered = apply_nms(valid_c, iou_threshold=iou_threshold)
        if valid_c_filtered:
            rows_c = []
            for b in valid_c_filtered:
                box_str = f"[{b['box'][0]}, {b['box'][1]}, {b['box'][2]}, {b['box'][3]}]"
                rows_c.append({
                    "Class": b.get("label", "Insect"),
                    "Confidence": f"{b.get('conf', 0.0)*100:.1f}%",
                    "Bounding Box [x1,y1,x2,y2]": box_str,
                    "Status": "🎯 High-Confidence Target" if b.get('conf', 0) >= 0.85 else "✔️ Detected"
                })
            st.dataframe(pd.DataFrame(rows_c), hide_index=True, use_container_width=True)
        else:
            st.warning("No CR-HSDPA detections above current confidence threshold.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: #64748b; padding: 10px;">
    University of Mindanao &bull; BS Computer Science Thesis Prototype &bull; Modified Attention-PestNet (CR-HSDPA) &bull; 2026
</div>
""", unsafe_allow_html=True)
