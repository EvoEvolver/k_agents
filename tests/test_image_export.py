"""Exercise the same image conversion used by the execution agent.

Chrome/Chromium must be installed, as in the deployment image.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest
from PIL import ImageStat

from k_agents.inspection.vlms import matplotlib_plotly_to_pil


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_visual_inspection_receives_a_rendered_image(backend):
    if backend == "plotly":
        figure = go.Figure(go.Scatter(y=[1, 3, 2]))
        figure.update_layout(width=400, height=300)
    else:
        figure, axes = plt.subplots(figsize=(4, 3), dpi=100)
        axes.plot([1, 3, 2])

    with matplotlib_plotly_to_pil(figure) as image:
        image.load()
        assert image.format == "PNG"
        assert image.size == (400, 300)
        assert max(ImageStat.Stat(image.convert("RGB")).var) > 0


def test_invalid_figure_is_rejected():
    with pytest.raises(ValueError, match="Matplotlib or Plotly"):
        matplotlib_plotly_to_pil(object())
