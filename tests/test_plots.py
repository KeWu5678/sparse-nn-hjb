"""Checks at the shared rendering interface, independent of paper artifacts."""

from io import BytesIO
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plots import (
    plot_model_value_surface,
    plot_normal_cross_section,
    plot_summary_frontier,
    plot_value_surface,
    save_figure,
)
from src.results import model_from_history, predict_model_physical


def _png(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=80)
    return buffer.getvalue()


def test_surface_legacy_wrapper_matches_prepared_renderer_and_does_not_clip_input():
    history = SimpleNamespace(
        best_iteration=0,
        inner_weights=[{"weight": np.array([[1., -2.]]), "bias": np.array([0.5])}],
        outer_weights=[np.array([[20.]])],
    )
    x0, x1 = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-2, 2, 7))
    value, _ = predict_model_physical(model_from_history(history),
                                      np.column_stack([x0.ravel(), x1.ravel()]))
    values = value.numpy().reshape(x0.shape)
    original = values.copy()
    assert original.max() > 60
    fig_old, _ = plot_model_value_surface(history, x_range=(-1, 1), y_range=(-2, 2),
                                         grid_n=7, vmax=60, show=False)
    fig_new, ax = plot_value_surface(x0, x1, values, clip=(0, 60))
    np.testing.assert_array_equal(values, original)
    assert ax.get_zlim() == (0, 60)
    assert _png(fig_old) == _png(fig_new)
    plt.close(fig_old)
    plt.close(fig_new)


def test_cross_section_keeps_supplied_reference_and_existing_style(tmp_path):
    x = np.array([-1., 0., 1.])
    reference = np.array([49.54, 25., 27.28])
    prediction = np.array([26.09, 25., 10.])
    fig, ax = plot_normal_cross_section(x, reference,
                                       [dict(y=prediction, color="red", ls="--", label="model")],
                                       ylabel="V")
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), reference)
    np.testing.assert_array_equal(ax.lines[2].get_ydata(), prediction)
    assert ax.lines[1].get_linewidth() == 2.6
    assert tuple(fig.get_size_inches()) == (8.5, 4.4)
    path = tmp_path / "nested" / "section.png"
    assert save_figure(fig, path, bbox_inches="tight") == [path]
    assert path.is_file()
    assert not plt.fignum_exists(fig.number)


def test_summary_frontier_preserves_distinct_marker_policy():
    fig, ax = plot_summary_frontier([dict(ns=np.arange(1, 49), h1=np.arange(48, 0, -1),
                                        color="red", ls="--", label="model")])
    assert tuple(fig.get_size_inches()) == (8.5, 6)
    assert ax.lines[0].get_markevery() == 4
    assert ax.lines[0].get_markersize() == 6
    assert ax.get_yscale() == "log"
    plt.close(fig)
