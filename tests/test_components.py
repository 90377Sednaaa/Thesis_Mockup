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
