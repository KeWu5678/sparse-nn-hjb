"""Checks at the shared rendering interface, independent of paper artifacts."""


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plots import (
    plot_feedback_trace,
    plot_summary_frontier,
    plot_value_surface,
    save_figure,
)


def test_surface_clips_the_display_without_changing_its_input():
    x0, x1 = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-2, 2, 7))
    values = 20.0 * np.tanh(x0 - 2.0 * x1 + 0.5) + 40.0
    original = values.copy()
    assert original.max() > 59
    fig, ax = plot_value_surface(x0, x1, values, clip=(0, 50))
    np.testing.assert_array_equal(values, original)
    assert ax.get_zlim() == (0, 50)
    plt.close(fig)


def test_save_figure_creates_folders_and_closes_the_figure(tmp_path):
    fig, _ = plt.subplots()
    path = tmp_path / "nested" / "figure.png"
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


def test_signed_control_trace_displays_negative_controls():
    control = np.array([-2., 1., -.5])
    fig, ax = plot_feedback_trace(
        [dict(t=np.arange(3), y=control, color="black", ls="-", lw=2, label="controller")],
        ylabel="u(t)", time_limit=2., nonnegative=False,
    )
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), control)
    assert ax.get_ylim()[0] <= -2.
    plt.close(fig)
