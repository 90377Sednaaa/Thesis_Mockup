"""
UI and Visualization Components for Thesis PestNet Mockup.
Provides publication-grade Pareto curve plots, clean architectural diagrams,
and engineered comparison cards for Streamlit rendering.
"""

import matplotlib.pyplot as plt
from config import ABLATION_DATA

def plot_ablation_pareto_curve():
    """
    Plots the GFLOPs vs mAP@50 Pareto frontier curve for the ablation variants.
    Styled as a publication-ready figure.
    """
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=140)
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#111726")

    variants = [d["variant"] for d in ABLATION_DATA]
    gflops = [d["gflops"] for d in ABLATION_DATA]
    map50 = [d["map50"] for d in ABLATION_DATA]
    ratios = [d["ratio"] for d in ABLATION_DATA]

    # Connect line
    ax.plot(gflops, map50, color="#475569", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)

    # Point formatting
    colors = []
    sizes = []
    for d in ABLATION_DATA:
        if d["variant"] == "CR-0.50":
            colors.append("#10b981")  # Selected Best (Emerald)
            sizes.append(190)
        elif "Baseline" in d["variant"]:
            colors.append("#f59e0b")  # Baseline (Amber)
            sizes.append(150)
        elif d["variant"] == "CR-0.25":
            colors.append("#ef4444")  # Aggressive feature loss
            sizes.append(130)
        else:
            colors.append("#38bdf8")  # Cyan / Blue
            sizes.append(130)

    ax.scatter(gflops, map50, c=colors, s=sizes, edgecolors="#ffffff", linewidths=1.5, zorder=2)

    # Annotations
    for i, d in enumerate(ABLATION_DATA):
        v = d["variant"]
        gf = d["gflops"]
        mp = d["map50"]
        
        if v == "CR-0.50":
            ax.annotate(
                f"CR-0.50 (Proposed Best)\n-27.2% GFLOPs  |  +0.73% mAP",
                (gf, mp),
                xytext=(gf - 20, mp + 0.36),
                color="#34d399",
                fontweight="600",
                fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="#34d399", lw=1.5),
                zorder=3
            )
        elif "Baseline" in v:
            ax.annotate(
                f"Baseline HSDPA (Reference)\n162.7 GFLOPs  |  68.67% mAP",
                (gf, mp),
                xytext=(gf - 24, mp - 0.58),
                color="#fbbf24",
                fontweight="600",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.3),
                zorder=3
            )
        elif v == "CR-0.25":
            ax.annotate(
                f"{v} (r=0.25)\nFeature Loss",
                (gf, mp),
                xytext=(gf + 1.5, mp - 0.45),
                color="#f87171",
                fontsize=8.5,
                zorder=3
            )
        else:
            ax.annotate(
                f"{v} (r={ratios[i]})",
                (gf, mp),
                xytext=(gf + 1.5, mp + 0.15),
                color="#cbd5e1",
                fontsize=8.5,
                zorder=3
            )

    # Guidelines
    ax.axhline(68.67, color="#d97706", linestyle=":", alpha=0.55, linewidth=1.2, label="Baseline mAP@50 (68.67%)")
    target_gflops = 162.70 * 0.90
    ax.axvline(target_gflops, color="#059669", linestyle=":", alpha=0.55, linewidth=1.2, label="Target Bound (≥10% GFLOPs Saved)")

    ax.set_title("Pareto Efficiency Curve: Computational Complexity vs. Detection Accuracy", color="#f8fafc", fontsize=11, pad=14, fontweight="600")
    ax.set_xlabel("Computational Complexity (GFLOPs) — Lower is Faster & Lighter", color="#94a3b8", fontsize=9.5, labelpad=8)
    ax.set_ylabel("Detection Precision (mAP@50 %) — Higher is Better", color="#94a3b8", fontsize=9.5, labelpad=8)

    ax.tick_params(colors="#94a3b8", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#222f46")

    ax.grid(True, linestyle=":", color="#1e293b", alpha=0.7)
    ax.legend(facecolor="#161f30", edgecolor="#2d3748", labelcolor="#cbd5e1", fontsize=8.5, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    return fig

def render_architecture_diagram_html() -> str:
    """
    Returns an engineered HTML pipeline comparison showing Baseline HSDPA vs Proposed CR-HSDPA.
    """
    html = """
    <div style="background: #0f1523; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 22px; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;">
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <!-- Baseline HSDPA -->
            <div style="flex: 1; min-width: 320px; background: #151b2a; border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 18px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
                    <div>
                        <div style="font-size: 14px; font-weight: 700; color: #f59e0b;">Baseline Attention-PestNet Neck</div>
                        <div style="font-size: 11px; color: #94a3b8;">Original HSDPA Module (Doan et al., 2026)</div>
                    </div>
                    <span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(245, 158, 11, 0.3);">Full Channels</span>
                </div>
                
                <div style="background: #0b0f19; border-radius: 6px; padding: 12px; font-size: 12.5px; line-height: 1.6; color: #94a3b8; border: 1px solid #1e293b;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>Input Feature Map:</span>
                        <code style="color: #f59e0b; background: #1c2233; padding: 1px 6px; border-radius: 3px;">Shape (B, C, H, W)</code>
                    </div>
                    <div style="background: #1a1e2e; border-radius: 6px; padding: 10px; margin: 8px 0; border: 1px dashed rgba(245, 158, 11, 0.4); text-align: center;">
                        <span style="color: #fbbf24; font-weight: 600; font-size: 12px;">4-Level Top-Down Attention (TDA)</span><br>
                        <span style="font-size: 11px; color: #94a3b8;">TDA₁ → TDA₂ → TDA₃ → TDA₄</span><br>
                        <span style="font-size: 10.5px; color: #f87171; font-weight: 500;">All 4 levels compute attention across full C channels</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                        <span>Forward Complexity:</span>
                        <strong style="color: #e2e8f0;">162.7 GFLOPs  |  79.66M params</strong>
                    </div>
                </div>

                <div style="margin-top: 12px; font-size: 12px; color: #fca5a5; background: rgba(239, 68, 68, 0.08); border-left: 3px solid #ef4444; padding: 8px 12px; border-radius: 0 4px 4px 0;">
                    <strong>Bottleneck:</strong> Calculating quadratic attention matrices on full channel width across repeated blocks causes high computation and latency.
                </div>
            </div>

            <!-- Proposed CR-HSDPA -->
            <div style="flex: 1; min-width: 320px; background: #151b2a; border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 18px; box-shadow: 0 0 15px rgba(16,185,129,0.06);">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
                    <div>
                        <div style="font-size: 14px; font-weight: 700; color: #10b981;">Proposed CR-HSDPA Neck (Our Model)</div>
                        <div style="font-size: 11px; color: #94a3b8;">Channel-Reduced Attention (r = 0.50)</div>
                    </div>
                    <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.3);">-27.2% GFLOPs</span>
                </div>

                <div style="background: #0b0f19; border-radius: 6px; padding: 12px; font-size: 12.5px; line-height: 1.6; color: #94a3b8; border: 1px solid #1e293b;">
                    <div style="background: rgba(16, 185, 129, 0.12); color: #6ee7b7; border-radius: 4px; padding: 5px 8px; margin-bottom: 8px; font-size: 11.5px; font-weight: 600; text-align: center;">
                        Step 1: 1×1 Conv + BatchNorm (Channel Compression C → 0.5C)
                    </div>
                    <div style="background: #112822; border-radius: 6px; padding: 10px; margin: 8px 0; border: 1px dashed rgba(16, 185, 129, 0.5); text-align: center;">
                        <span style="color: #34d399; font-weight: 600; font-size: 12px;">Compressed 4-Level TDA Hierarchy</span><br>
                        <span style="font-size: 11px; color: #a7f3d0;">TDA₁(0.5C) → TDA₂(0.5C) → TDA₃(0.5C) → TDA₄(0.5C)</span><br>
                        <span style="font-size: 10.5px; color: #6ee7b7; font-weight: 500;">Attention runs on 50% fewer channels with feature filtering</span>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.12); color: #6ee7b7; border-radius: 4px; padding: 5px 8px; margin-top: 8px; font-size: 11.5px; font-weight: 600; text-align: center;">
                        Step 2: Concat + 1×1 Conv + BatchNorm (Channel Restoration 0.5C → C)
                    </div>
                </div>

                <div style="margin-top: 12px; font-size: 12px; color: #6ee7b7; background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 8px 12px; border-radius: 0 4px 4px 0;">
                    <strong>Advantage:</strong> Eliminates channel redundancy to save -27.2% compute, achieves 53.8 FPS real-time speed, and suppresses noisy leaf textures.
                </div>
            </div>
        </div>
    </div>
    """
    return html

def render_metric_card(label: str, baseline_val: str, cr_val: str, delta: str, is_positive: bool = True) -> str:
    """Renders a styled comparison card HTML string."""
    badge_bg = "rgba(16, 185, 129, 0.15)" if is_positive else "rgba(239, 68, 68, 0.15)"
    badge_color = "#34d399" if is_positive else "#fca5a5"
    border_color = "rgba(16, 185, 129, 0.3)" if is_positive else "rgba(239, 68, 68, 0.3)"
    
    return f"""
    <div style="background: #111726; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 14px; margin-bottom: 10px;">
        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">{label}</div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">
            <div>
                <span style="font-size: 18px; font-weight: 700; color: #f8fafc; font-family: ui-monospace, 'SF Mono', Consolas, monospace;">{cr_val}</span>
                <span style="font-size: 12px; color: #64748b; margin-left: 6px;">vs {baseline_val}</span>
            </div>
            <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {border_color}; font-size: 11px; font-weight: 600; padding: 2px 7px; border-radius: 4px; font-family: ui-monospace, 'SF Mono', Consolas, monospace;">
                {delta}
            </span>
        </div>
    </div>
    """
