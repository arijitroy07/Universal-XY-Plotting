import numpy as np

from slope_tools import calculate_slope_segment


def test_x_range_fit_and_signed_differences():
    x = np.arange(-5.0, 6.0)
    y = 2.0 * x - 3.0

    result = calculate_slope_segment(x, y, "X axis", -2.0, 4.0)

    assert np.isclose(result.x_start, -2.0)
    assert np.isclose(result.x_end, 4.0)
    assert np.isclose(result.delta_x, 6.0)
    assert np.isclose(result.y_start, -7.0)
    assert np.isclose(result.y_end, 5.0)
    assert np.isclose(result.delta_y, 12.0)


def test_reversed_range_preserves_negative_differences():
    x = np.arange(-5.0, 6.0)
    y = -1.5 * x + 4.0

    result = calculate_slope_segment(x, y, "X axis", 4.0, -2.0)

    assert np.isclose(result.delta_x, -6.0)
    assert np.isclose(result.delta_y, 9.0)


def test_y_range_and_axis_clipping_adjust_output_coordinates():
    x = np.arange(0.0, 11.0)
    y = 3.0 * x + 1.0

    result = calculate_slope_segment(
        x,
        y,
        "Y axis",
        4.0,
        25.0,
        visible_x_min=2.0,
        visible_x_max=6.0,
    )

    assert np.isclose(result.x_start, 2.0)
    assert np.isclose(result.y_start, 7.0)
    assert np.isclose(result.x_end, 6.0)
    assert np.isclose(result.y_end, 19.0)
    assert np.isclose(result.delta_x, 4.0)
    assert np.isclose(result.delta_y, 12.0)
