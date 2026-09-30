"""
UI and Visualization Components for Thesis PestNet Mockup.
Provides Pareto curve ablation plots, architectural visual diagrams,
and styled metric cards for Streamlit rendering.
"""

import matplotlib.pyplot as plt
from config import ABLATION_DATA

def plot_ablation_pareto_curve():
    """
    Plots the GFLOPs vs mAP@50 Pareto curve for the ablation configurations.
    Returns a matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    variants = [d["variant"] for d in ABLATION_DATA]
    gflops = [d["gflops"] for d in ABLATION_DATA]
    map50 = [d["map50"] for d in ABLATION_DATA]
    ratios = [d["ratio"] for d in ABLATION_DATA]

    # Plot line connect
    ax.plot(gflops, map50, color="#64748b", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)

    # Plot points
    colors = []
    sizes = []
    for d in ABLATION_DATA:
        if d["variant"] == "CR-0.50":
            colors.append("#10b981")  # Selected Best (Emerald)
            sizes.append(180)
        elif "Baseline" in d["variant"]:
            colors.append("#f59e0b")  # Baseline (Amber)
            sizes.append(150)
        elif d["variant"] == "CR-0.25":
            colors.append("#ef4444")  # Aggressive feature loss (Red)
            sizes.append(120)
        else:
            colors.append("#3b82f6")  # Blue
            sizes.append(120)

    scatter = ax.scatter(gflops, map50, c=colors, s=sizes, edgecolors="#ffffff", linewidths=1.5, zorder=2)

    # Annotate points
    for i, d in enumerate(ABLATION_DATA):
        v = d["variant"]
        gf = d["gflops"]
        mp = d["map50"]
        offset_y = 0.25 if v != "CR-0.25" else -0.45
        offset_x = 0.5
        
        if v == "CR-0.50":
            label = f"★ {v} (Our Best Model)\n-27.2% GFLOPs, +0.73% mAP"
            ax.annotate(
                label,
                (gf, mp),
                xytext=(gf - 18, mp + 0.35),
                color="#34d399",
                fontweight="bold",
                fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color="#34d399", lw=1.5),
                zorder=3
            )
        elif "Baseline" in v:
            ax.annotate(
                f"{v} (Original)\n{gf} GF, {mp}%",
                (gf, mp),
                xytext=(gf - 16, mp - 0.55),
                color="#fbbf24",
                fontweight="bold",
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.2),
                zorder=3
            )
        else:
            ax.annotate(
                f"{v} ({ratios[i]})",
                (gf, mp),
                xytext=(gf + offset_x, mp + offset_y),
                color="#cbd5e1",
                fontsize=8,
                zorder=3
            )

    # Threshold guidelines
    ax.axhline(68.67, color="#f59e0b", linestyle=":", alpha=0.5, label="Baseline mAP@50 (68.67%)")
    target_gflops = 162.70 * 0.90  # 10% reduction line
    ax.axvline(target_gflops, color="#10b981", linestyle=":", alpha=0.5, label="Target (≥10% GFLOPs Saved)")

    ax.set_title("Model Speed & Accuracy Trade-off (Ablation Results)", color="#f8fafc", fontsize=11, pad=12, fontweight="bold")
    ax.set_xlabel("Complexity (GFLOPs) — Lower is Faster & Lighter", color="#94a3b8", fontsize=9.5)
    ax.set_ylabel("Detection Accuracy (mAP@50 %) — Higher is Better", color="#94a3b8", fontsize=9.5)

    ax.tick_params(colors="#94a3b8", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(True, linestyle=":", color="#334155", alpha=0.6)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8, loc="lower left")

    plt.tight_layout()
    return fig

def render_architecture_diagram_html() -> str:
    """
    Returns an HTML comparison illustrating the Baseline HSDPA vs Proposed CR-HSDPA neck block.
    """
    html = """
    <div style="background: #131720; border-radius: 10px; border: 1px solid #1e293b; padding: 20px; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <!-- Baseline HSDPA -->
            <div style="flex: 1; min-width: 300px; background: #1a1f2c; border: 1px solid #b45309; border-radius: 8px; padding: 16px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-weight: 700; color: #f59e0b; font-size: 14px;">Baseline Attention-PestNet</span>
                    <span style="background: #78350f; color: #fef3c7; font-size: 11px; padding: 2px 8px; border-radius: 4px;">Full Channels</span>
                </div>
                <div style="background: #0f131a; border-radius: 6px; padding: 12px; font-size: 12.5px; line-height: 1.6; color: #94a3b8;">
                    <div style="margin-bottom: 8px;">
                        <strong style="color: #e2e8f0;">Input:</strong> Full channels <code style="color: #f59e0b;">(C)</code>
                    </div>
                    <div style="background: #232a3b; border-radius: 6px; padding: 10px; margin: 8px 0; border: 1px dashed #d97706; text-align: center;">
                        <span style="color: #fbbf24; font-weight: 600;">4-Level Attention Processing</span><br>
                        <span style="font-size: 11px; color: #f87171;">⚠️ Heavy computation: Runs on all C channels</span>
                    </div>
                    <div>
                        <strong style="color: #e2e8f0;">Result:</strong> 162.7 GFLOPs, 79.6M parameters
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 11.5px; color: #fca5a5; background: rgba(239,68,68,0.12); padding: 8px 12px; border-radius: 4px;">
                    <strong>Drawback:</strong> Slower inference speed and higher memory usage.
                </div>
            </div>

            <!-- Proposed CR-HSDPA -->
            <div style="flex: 1; min-width: 300px; background: #1a1f2c; border: 1px solid #059669; border-radius: 8px; padding: 16px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span style="font-weight: 700; color: #10b981; font-size: 14px;">Our Model (CR-HSDPA)</span>
                    <span style="background: #064e3b; color: #a7f3d0; font-size: 11px; padding: 2px 8px; border-radius: 4px;">r = 0.50</span>
                </div>
                <div style="background: #0f131a; border-radius: 6px; padding: 12px; font-size: 12.5px; line-height: 1.6; color: #94a3b8;">
                    <div style="margin-bottom: 6px;">
                        <strong style="color: #e2e8f0;">1. Pre-Attention 1×1 Conv:</strong> Compresses channels from <code style="color: #34d399;">C → 0.5C</code>
                    </div>
                    <div style="background: #1e3a34; border-radius: 6px; padding: 10px; margin: 8px 0; border: 1px dashed #10b981; text-align: center;">
                        <span style="color: #34d399; font-weight: 600;">4-Level Attention on Half Channels</span><br>
                        <span style="font-size: 11px; color: #a7f3d0;">✔ -27.2% computation savings</span>
                    </div>
                    <div>
                        <strong style="color: #e2e8f0;">2. Post-Attention 1×1 Conv:</strong> Restores channels back to C for smooth detection
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 11.5px; color: #6ee7b7; background: rgba(16,185,129,0.12); padding: 8px 12px; border-radius: 4px;">
                    <strong>Advantage:</strong> Cuts computation by -27.2%, runs 53% faster, and improves mAP by removing noise.
                </div>
            </div>
        </div>
    </div>
    """
    return html

def render_metric_card(label: str, baseline_val: str, cr_val: str, delta: str, is_positive: bool = True) -> str:
    """Renders a styled comparison card HTML string."""
    badge_bg = "#064e3b" if is_positive else "#7f1d1d"
    badge_color = "#34d399" if is_positive else "#fca5a5"
    
    return f"""
    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
        <div style="font-size: 11px; color: #8b949e; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">{label}</div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">
            <div>
                <span style="font-size: 18px; font-weight: 700; color: #f0f6fc;">{cr_val}</span>
                <span style="font-size: 12px; color: #8b949e; margin-left: 6px;">vs {baseline_val}</span>
            </div>
            <span style="background: {badge_bg}; color: {badge_color}; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 12px;">
                {delta}
            </span>
        </div>
    </div>
    """
