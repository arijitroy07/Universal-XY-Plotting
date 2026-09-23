"""Linear slope fitting, clipping, plotting, and output utilities."""

from dataclasses import asdict, dataclass

import numpy as np


class SlopeCalculationError(ValueError):
    """Raised when a requested slope segment cannot be calculated."""


@dataclass(frozen=True)
class SlopeResult:
    """Coordinates and fitted-line information for one slope segment."""

    x_start: float
    x_end: float
    delta_x: float
    y_start: float
    y_end: float
    delta_y: float
    slope: float
    intercept: float
    points_used: int

    def to_dict(self):
        """Return a session-state-friendly dictionary."""
        return asdict(self)


def _finite_xy(x_values, y_values):
    """Return matching, finite, one-dimensional X and Y arrays."""
    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    y_values = np.asarray(y_values, dtype=float).reshape(-1)

    if x_values.size != y_values.size:
        raise SlopeCalculationError("The selected X and Y columns have different lengths.")

    finite = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[finite], y_values[finite]


def _ordered_plot_bounds(first, second):
    """Convert optional, possibly reversed axis limits into low/high bounds."""
    if first is not None and second is not None:
        return min(first, second), max(first, second)
    return first, second


def _clip_segment(
    x_start,
    y_start,
    x_end,
    y_end,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
):
    """Clip a directed line segment to optional visible plot boundaries."""
    x_low, x_high = _ordered_plot_bounds(x_min, x_max)
    y_low, y_high = _ordered_plot_bounds(y_min, y_max)

    dx = x_end - x_start
    dy = y_end - y_start
    t_enter = 0.0
    t_exit = 1.0

    def constrain(value_start, difference, lower, upper, current_enter, current_exit):
        if difference == 0:
            if lower is not None and value_start < lower:
                return None
            if upper is not None and value_start > upper:
                return None
            return current_enter, current_exit

        if lower is not None:
            boundary_t = (lower - value_start) / difference
            if difference > 0:
                current_enter = max(current_enter, boundary_t)
            else:
                current_exit = min(current_exit, boundary_t)

        if upper is not None:
            boundary_t = (upper - value_start) / difference
            if difference > 0:
                current_exit = min(current_exit, boundary_t)
            else:
                current_enter = max(current_enter, boundary_t)

        if current_enter > current_exit:
            return None

        return current_enter, current_exit

    clipped = constrain(x_start, dx, x_low, x_high, t_enter, t_exit)
    if clipped is None:
        raise SlopeCalculationError(
            "The requested slope does not intersect the visible X-axis range."
        )
    t_enter, t_exit = clipped

    clipped = constrain(y_start, dy, y_low, y_high, t_enter, t_exit)
    if clipped is None:
        raise SlopeCalculationError(
            "The requested slope does not intersect the visible Y-axis range."
        )
    t_enter, t_exit = clipped

    clipped_x_start = x_start + t_enter * dx
    clipped_y_start = y_start + t_enter * dy
    clipped_x_end = x_start + t_exit * dx
    clipped_y_end = y_start + t_exit * dy

    return clipped_x_start, clipped_y_start, clipped_x_end, clipped_y_end


def calculate_slope_segment(
    x_values,
    y_values,
    range_axis,
    range_start,
    range_end,
    visible_x_min=None,
    visible_x_max=None,
    visible_y_min=None,
    visible_y_max=None,
):
    """
    Fit Y = slope*X + intercept inside an X or Y selection range.

    The fitted segment follows the user's start-to-end direction. It is then
    clipped to customized visible axis limits, so its reported signed
    differences match the segment that is actually drawn.
    """
    x_values, y_values = _finite_xy(x_values, y_values)

    if x_values.size < 2:
        raise SlopeCalculationError("At least two finite X-Y points are required.")

    if range_start == range_end:
        raise SlopeCalculationError("The slope range start and end cannot be equal.")

    lower_range = min(range_start, range_end)
    upper_range = max(range_start, range_end)
    normalized_axis = range_axis.strip().lower()

    if normalized_axis.startswith("x"):
        selected = (x_values >= lower_range) & (x_values <= upper_range)
    elif normalized_axis.startswith("y"):
        selected = (y_values >= lower_range) & (y_values <= upper_range)
    else:
        raise SlopeCalculationError("Range axis must be either X axis or Y axis.")

    selected_x = x_values[selected]
    selected_y = y_values[selected]

    if selected_x.size < 2:
        raise SlopeCalculationError(
            "The selected range contains fewer than two data points."
        )

    if np.unique(selected_x).size < 2:
        raise SlopeCalculationError(
            "The selected range must contain at least two different X values."
        )

    slope, intercept = np.polyfit(selected_x, selected_y, 1)
    slope = float(slope)
    intercept = float(intercept)

    if normalized_axis.startswith("x"):
        x_start = float(range_start)
        x_end = float(range_end)
        y_start = slope * x_start + intercept
        y_end = slope * x_end + intercept
    else:
        tolerance = np.finfo(float).eps * max(1.0, abs(intercept))
        if abs(slope) <= tolerance:
            raise SlopeCalculationError(
                "A Y-defined segment cannot be calculated because the fitted slope is zero."
            )
        y_start = float(range_start)
        y_end = float(range_end)
        x_start = (y_start - intercept) / slope
        x_end = (y_end - intercept) / slope

    x_start, y_start, x_end, y_end = _clip_segment(
        x_start,
        y_start,
        x_end,
        y_end,
        x_min=visible_x_min,
        x_max=visible_x_max,
        y_min=visible_y_min,
        y_max=visible_y_max,
    )

    return SlopeResult(
        x_start=float(x_start),
        x_end=float(x_end),
        delta_x=float(x_end - x_start),
        y_start=float(y_start),
        y_end=float(y_end),
        delta_y=float(y_end - y_start),
        slope=slope,
        intercept=intercept,
        points_used=int(selected_x.size),
    )


def add_slope_line(
    axis,
    slope_result,
    label="Slope fit",
    color="#D62728",
    line_width=2.0,
    line_style="--",
    display_y_values=None,
):
    """Draw a fitted slope segment on a Matplotlib axis."""
    if display_y_values is None:
        plotted_y = [slope_result.y_start, slope_result.y_end]
    else:
        plotted_y = display_y_values

    axis.plot(
        [slope_result.x_start, slope_result.x_end],
        plotted_y,
        label=label,
        color=color,
        linewidth=line_width,
        linestyle=line_style,
        zorder=10,
    )
