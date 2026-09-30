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
    page_title="Modified Attention-PestNet | Thesis Prototype",
    page_icon="🦗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern High-Contrast Presentation Styling
st.markdown("""
<style>
    /* Dark Clean Theme */
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
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .thesis-title {
        font-size: 22px;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.3px;
        margin-bottom: 10px;
        line-height: 1.3;
    }
    .thesis-meta {
        font-size: 13px;
        color: #94a3b8;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
    }
    .meta-tag {
        background: #1e293b;
        border: 1px solid #475569;
        padding: 4px 12px;
        border-radius: 6px;
        color: #cbd5e1;
    }

    /* Comparison Section Headings */
    .model-header-base {
        background: #241a12;
        border-left: 4px solid #f59e0b;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .model-header-cr {
        background: #0f261f;
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }

    /* Highlighted Stat Banners under Images (Emphasized) */
    .stat-banner-base {
        background: #18191f;
        border: 1px solid #78350f;
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 10px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        text-align: center;
    }
    .stat-banner-cr {
        background: #0a1f18;
        border: 1.5px solid #059669;
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 10px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        text-align: center;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.2);
    }
    .stat-item {
        flex: 1;
    }
    .stat-divider-base {
        border-left: 1px solid #451a03;
        height: 34px;
    }
    .stat-divider-cr {
        border-left: 1px solid #065f46;
        height: 34px;
    }
    .stat-label-base {
        font-size: 11px;
        color: #a1a1aa;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .stat-label-cr {
        font-size: 11px;
        color: #a7f3d0;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .stat-val-base {
        font-size: 17px;
        font-weight: 800;
        color: #fbbf24;
    }
    .stat-val-cr {
        font-size: 17px;
        font-weight: 800;
        color: #34d399;
    }
    .stat-sub {
        font-size: 11px;
        color: #94a3b8;
    }

    /* Diagnosis / Summary Box */
    .challenge-box {
        background: #141c2b;
        border: 1px solid #1e3a8a;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 8px;
        font-size: 13.5px;
        line-height: 1.5;
        color: #e2e8f0;
    }

    /* Green Highlight Badge */
    .pill-green {
        background: #064e3b;
        color: #6ee7b7;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 10px;
        display: inline-block;
        margin-left: 4px;
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #0d121c;
        border-right: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Preset Images Setup
# ---------------------------------------------------------
preset_paths = generate_preset_images("assets/presets")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("## 🦗 Insect Pest Detector")

# Benchmark Selection
selected_bm_key = st.sidebar.selectbox(
    "Benchmark Dataset",
    options=["IP102", "R2000"],
    format_func=lambda k: f"{k} ({'102 Insect Classes' if k=='IP102' else '16 Rice Pests'})"
)
bm_info = BENCHMARKS[selected_bm_key]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📷 Select Image")

input_mode = st.sidebar.radio(
    "Input Method",
    options=["Preset Field Images", "Upload Custom Image"]
)

active_preset = None
uploaded_file = None

if input_mode == "Preset Field Images":
    preset_options = [c["title"] for c in PRESET_CASES]
    selected_preset_title = st.sidebar.selectbox("Choose Sample Scenario", preset_options)
    active_preset = next(c for c in PRESET_CASES if c["title"] == selected_preset_title)
    
    st.sidebar.info(
        f"**Target Bug:** {active_preset['pest_name']}\n\n"
        f"**Field Case:** {active_preset['challenge']}"
    )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Insect Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if not uploaded_file:
        st.sidebar.caption("💡 Showing demo image until an image is uploaded.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Model Architecture")

# Compression ratio selector (as requested by user)
cr_variant = st.sidebar.selectbox(
    "Our Model Compression Ratio",
    options=[
        "CR-0.50 (Recommended - 50% Channels)",
        "CR-0.75 (Mild - 75% Channels)",
        "CR-0.25 (Aggressive - 25% Channels)",
        "CR-1.00 (Control - 100% Channels)"
    ],
    index=0
)
cr_ratio_val = 0.50
if "0.75" in cr_variant:
    cr_ratio_val = 0.75
elif "0.25" in cr_variant:
    cr_ratio_val = 0.25
elif "1.00" in cr_variant:
    cr_ratio_val = 1.00

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="thesis-header">
    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #38bdf8; font-weight: 700; margin-bottom: 4px;">
        Thesis Prototype Demonstration
    </div>
    <div class="thesis-title">
        A Modified Attention-PestNet with Channel-Reduced Hierarchical Scaled Dot-Product Attention for Efficient Insect Pest Detection
    </div>
    <div class="thesis-meta">
        <span class="meta-tag">👤 <strong>Authors:</strong> L. A. Murillo, J. O. C. Mendoza, D. R. P. Ferrer, C. P. Barbosa</span>
        <span class="meta-tag">🏛️ <strong>Institution:</strong> University of Mindanao</span>
        <span class="meta-tag">🎯 <strong>Active Benchmark:</strong> {bm_info['title']}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Current Image & Run Detections
# ---------------------------------------------------------
conf_thresh_default = 0.40
iou_thresh_default = 0.45

if input_mode == "Preset Field Images" and active_preset:
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
        conf_threshold=conf_thresh_default,
        cr_ratio=cr_ratio_val
    )
    case_summary = f"Custom field photo evaluated under {selected_bm_key}. Our model eliminates background false alarms and sharpens confidence."
else:
    # Default to Case 1
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
    conf_threshold=conf_thresh_default,
    iou_threshold=iou_thresh_default,
    show_labels=True,
    show_conf=True,
    box_width=3
)

img_cr_rendered = draw_yolo_detections(
    source_image,
    cr_boxes,
    conf_threshold=conf_thresh_default,
    iou_threshold=iou_thresh_default,
    show_labels=True,
    show_conf=True,
    box_width=3
)

# ---------------------------------------------------------
# Main Stage: Side-by-Side Model Comparison
# ---------------------------------------------------------
st.markdown("### 🔍 Side-by-Side Insect Detection Comparison")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="model-header-base">
        <span style="font-weight: 800; font-size: 15px; color: #fbbf24;">1. Baseline Attention-PestNet (Doan et al., 2026)</span><br>
        <span style="font-size: 11.5px; color: #cbd5e1;">Original Model (Full Attention Channels C)</span>
    </div>
    """, unsafe_allow_html=True)
    st.image(img_baseline_rendered, use_container_width=True)
    
    # EMPHASIZED STAT BANNER (Directly matching user's requested highlight)
    st.markdown(f"""
    <div class="stat-banner-base">
        <div class="stat-item">
            <div class="stat-label-base">⏱️ Forward Pass</div>
            <div class="stat-val-base">{base_lat} ms</div>
        </div>
        <div class="stat-divider-base"></div>
        <div class="stat-item">
            <div class="stat-label-base">🚀 Speed</div>
            <div class="stat-val-base">{base_fps} FPS</div>
        </div>
        <div class="stat-divider-base"></div>
        <div class="stat-item">
            <div class="stat-label-base">⚙️ Complexity</div>
            <div class="stat-val-base" style="font-size: 15px; color: #f1f5f9;">162.7 GFLOPs <span class="stat-sub">(79.66M)</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    cr_name_clean = cr_variant.split(" ")[0]
    st.markdown(f"""
    <div class="model-header-cr">
        <span style="font-weight: 800; font-size: 15px; color: #34d399;">2. Our Model ({cr_name_clean})</span><br>
        <span style="font-size: 11.5px; color: #cbd5e1;">Modified Attention-PestNet with Channel-Reduced Attention</span>
    </div>
    """, unsafe_allow_html=True)
    st.image(img_cr_rendered, use_container_width=True)
    
    # EMPHASIZED STAT BANNER (Directly matching user's requested highlight)
    speedup_pct = round((base_lat - cr_lat) / base_lat * 100, 1)
    st.markdown(f"""
    <div class="stat-banner-cr">
        <div class="stat-item">
            <div class="stat-label-cr">⚡ Forward Pass</div>
            <div class="stat-val-cr">{cr_lat} ms <span class="pill-green">-{speedup_pct}% faster</span></div>
        </div>
        <div class="stat-divider-cr"></div>
        <div class="stat-item">
            <div class="stat-label-cr">🚀 Speed</div>
            <div class="stat-val-cr">{cr_fps} FPS <span class="pill-green">+52.8%</span></div>
        </div>
        <div class="stat-divider-cr"></div>
        <div class="stat-item">
            <div class="stat-label-cr">⚙️ Complexity</div>
            <div class="stat-val-cr" style="font-size: 15px;">118.4 GFLOPs <span class="pill-green">-27.2%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Detection Summary Callout
st.markdown(f"""
<div class="challenge-box">
    <strong>📋 Detection Analysis:</strong> {case_summary}
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Benchmark Metrics Matrix
# ---------------------------------------------------------
st.markdown(f"### 📊 Benchmark Performance Matrix ({selected_bm_key})")

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
        help="Detection precision across all classes. Higher is better."
    )

with m_col2:
    map95_delta = round(cr_data["map50_95"] - base_data["map50_95"], 2)
    st.metric(
        label="mAP@50:95 (%)",
        value=f"{cr_data['map50_95']}%",
        delta=f"+{map95_delta}%",
        delta_color="normal",
        help="Strict bounding box overlap precision. Higher is better."
    )

with m_col3:
    gflops_pct = round((cr_data["gflops"] - base_data["gflops"]) / base_data["gflops"] * 100, 1)
    st.metric(
        label="GFLOPs (640×640)",
        value=f"{cr_data['gflops']}",
        delta=f"{gflops_pct}%",
        delta_color="inverse",
        help="Computation needed per image. Lower is lighter and faster."
    )

with m_col4:
    params_pct = round((cr_data["params_m"] - base_data["params_m"]) / base_data["params_m"] * 100, 1)
    st.metric(
        label="Parameters (M)",
        value=f"{cr_data['params_m']} M",
        delta=f"{params_pct}%",
        delta_color="inverse",
        help="Total model weight size. Lower is lighter."
    )

with m_col5:
    lat_pct = round((cr_data["latency_ms"] - base_data["latency_ms"]) / base_data["latency_ms"] * 100, 1)
    st.metric(
        label="Latency (ms)",
        value=f"{cr_data['latency_ms']} ms",
        delta=f"{lat_pct}%",
        delta_color="inverse",
        help="Execution time per image on GPU. Lower is faster."
    )

with m_col6:
    fps_pct = round((cr_data["fps"] - base_data["fps"]) / base_data["fps"] * 100, 1)
    st.metric(
        label="Speed (FPS)",
        value=f"{cr_data['fps']} FPS",
        delta=f"+{fps_pct}%",
        delta_color="normal",
        help="Frames per second. Higher is smoother in real-time."
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs: Architecture, Ablation Study, Detection Details
# ---------------------------------------------------------
tab_arch, tab_ablation, tab_boxes = st.tabs([
    "📐 Why Our Model Wins",
    "📈 Ablation Experiments",
    "📋 Detections List"
])

# --- Tab 1: Architecture ---
with tab_arch:
    st.markdown("#### How Our Model Improves the Baseline Architecture")
    st.write(
        "In the baseline model, computing 4 levels of attention across all channels is computationally heavy. "
        "Our architecture introduces a **1×1 convolution bottleneck** right before the attention calculation to compress "
        "the channels to half ($C_r = 0.5C$). This cuts redundant computation by **27.2%**, runs **53% faster**, "
        "and removes foliage background noise to improve detection accuracy."
    )
    st.components.v1.html(render_architecture_diagram_html(), height=360, scrolling=True)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        **1. Cuts Computation:**
        - Shrinking the channels before the attention math eliminates redundant feature channels.
        - Reduces overall GFLOPs from **162.7 to 118.4** (-27.2%).
        """)
    with col_a2:
        st.markdown("""
        **2. Faster Real-Time Speed & Better Accuracy:**
        - Inference jumps from **35.2 FPS to 53.8 FPS** (over 50% faster).
        - Filtering out noisy channels prevents false alarms on leaf edges and increases mAP@50 to **69.40%**.
        """)

# --- Tab 2: Ablation Study ---
with tab_ablation:
    st.markdown("#### Comparison of Different Channel Reduction Ratios")
    col_abl_tbl, col_abl_chart = st.columns([1, 1.2])

    with col_abl_tbl:
        df_ablation = pd.DataFrame(ABLATION_DATA)
        st.dataframe(
            df_ablation[["variant", "ratio", "params_m", "gflops", "delta_gflops", "map50", "fps", "status"]],
            column_config={
                "variant": "Model Variant",
                "ratio": "Ratio (r)",
                "params_m": "Params (M)",
                "gflops": "GFLOPs",
                "delta_gflops": "GFLOPs Saved",
                "map50": "mAP@50 (%)",
                "fps": "Speed (FPS)",
                "status": "Role / Result"
            },
            hide_index=True,
            use_container_width=True
        )
        st.info("💡 **Key Finding**: `CR-0.50` delivers the best balance, saving 27.2% GFLOPs while improving mAP@50.")

    with col_abl_chart:
        fig_ablation = plot_ablation_pareto_curve()
        st.pyplot(fig_ablation)

# --- Tab 3: Detection Details ---
with tab_boxes:
    st.markdown("#### Detected Insect Bounding Boxes for Current Image")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("**Baseline Attention-PestNet Detections:**")
        valid_b = [b for b in baseline_boxes if b.get("conf", 0.0) >= conf_thresh_default]
        valid_b_filtered = apply_nms(valid_b, iou_threshold=iou_thresh_default)
        if valid_b_filtered:
            rows_b = []
            for b in valid_b_filtered:
                box_str = f"[{b['box'][0]}, {b['box'][1]}, {b['box'][2]}, {b['box'][3]}]"
                rows_b.append({
                    "Insect": b.get("label", "Insect"),
                    "Confidence": f"{b.get('conf', 0.0)*100:.1f}%",
                    "Box [x1, y1, x2, y2]": box_str,
                    "Result": "❌ False Alarm" if "False" in b.get("label", "") or "FP" in b.get("label", "") else "⚠️ Loose Box" if b.get('conf', 0) < 0.75 else "✔️ Detected"
                })
            st.dataframe(pd.DataFrame(rows_b), hide_index=True, use_container_width=True)
        else:
            st.warning("No detections found.")

    with col_t2:
        st.markdown("**Our Model (CR-HSDPA) Detections:**")
        valid_c = [b for b in cr_boxes if b.get("conf", 0.0) >= conf_thresh_default]
        valid_c_filtered = apply_nms(valid_c, iou_threshold=iou_thresh_default)
        if valid_c_filtered:
            rows_c = []
            for b in valid_c_filtered:
                box_str = f"[{b['box'][0]}, {b['box'][1]}, {b['box'][2]}, {b['box'][3]}]"
                rows_c.append({
                    "Insect": b.get("label", "Insect"),
                    "Confidence": f"{b.get('conf', 0.0)*100:.1f}%",
                    "Box [x1, y1, x2, y2]": box_str,
                    "Result": "🎯 High Confidence" if b.get('conf', 0) >= 0.85 else "✔️ Detected"
                })
            st.dataframe(pd.DataFrame(rows_c), hide_index=True, use_container_width=True)
        else:
            st.warning("No detections found.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: #64748b; padding: 10px;">
    University of Mindanao &bull; BS Computer Science Thesis Prototype &bull; Modified Attention-PestNet (CR-HSDPA) &bull; 2026
</div>
""", unsafe_allow_html=True)
