"""
Slope calculation, branch selection, axis clipping, and plotting utilities.

X-defined range:
    Selects points using their X coordinates and fits Y as a function of X.

Y-defined range:
    Identifies continuous curve branches connecting the requested Y values.
    The selected branch is fitted as X as a function of Y. This is more stable
    for vertical and nearly vertical data than fitting Y as a function of X.
"""

from dataclasses import asdict, dataclass

import numpy as np


class SlopeCalculationError(ValueError):
    """Raised when a requested slope segment cannot be calculated."""


@dataclass(frozen=True)
class SlopeResult:
    """
    Store the coordinates and fitted-line information for one slope segment.

    The start and end coordinates are the points actually displayed after
    clipping against customized X-axis or Y-axis limits.
    """

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
        """Return a dictionary suitable for Streamlit session state."""
        return asdict(self)


def _finite_xy(x_values, y_values):
    """
    Convert X and Y to one-dimensional arrays and remove nonfinite pairs.

    A point is retained only when both its X value and Y value are finite.
    """

    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    y_values = np.asarray(y_values, dtype=float).reshape(-1)

    if x_values.size != y_values.size:
        raise SlopeCalculationError(
            "The selected X and Y columns have different lengths."
        )

    finite = np.isfinite(x_values) & np.isfinite(y_values)

    return x_values[finite], y_values[finite]


def _ordered_plot_bounds(first, second):
    """
    Convert optional axis limits into lower and upper numerical boundaries.

    This is used only for clipping calculations. The original entered order
    remains unchanged in app.py, allowing a reversed X-axis.
    """

    if first is not None and second is not None:
        return min(first, second), max(first, second)

    return first, second


def _crossing_positions(values, target):
    """
    Find every row position where a data curve crosses a target value.

    A crossing can occur between two measured rows. Such a crossing is stored
    as a fractional row position. For example, 10.5 means the crossing is
    halfway between rows 10 and 11.
    """

    values = np.asarray(values, dtype=float)
    positions = []

    # Numerical tolerance prevents tiny floating-point differences from
    # preventing recognition of a value that is effectively equal.
    tolerance = np.finfo(float).eps * 100 * max(
        1.0,
        abs(float(target)),
        float(np.max(np.abs(values))),
    )

    for index in range(values.size - 1):
        first = float(values[index])
        second = float(values[index + 1])

        # Record an exact or numerically equivalent measured point.
        if np.isclose(
            first,
            target,
            rtol=0.0,
            atol=tolerance,
        ):
            positions.append(float(index))

        first_offset = first - target
        second_offset = second - target

        # Opposite signs mean that the target lies between these two rows.
        if first_offset * second_offset < 0:
            fraction = (target - first) / (second - first)
            positions.append(float(index + fraction))

    # The final row is not the first point of another line segment, so it must
    # be checked separately.
    if np.isclose(
        values[-1],
        target,
        rtol=0.0,
        atol=tolerance,
    ):
        positions.append(float(values.size - 1))

    # Remove duplicate crossing positions, which can occur when a measured
    # point is exactly equal to the requested boundary.
    unique_positions = []

    for position in sorted(positions):
        if not unique_positions or not np.isclose(
            position,
            unique_positions[-1],
            rtol=0.0,
            atol=1e-10,
        ):
            unique_positions.append(position)

    return unique_positions


def _interpolate_xy_at_position(
    x_values,
    y_values,
    position,
):
    """
    Interpolate X and Y coordinates at a fractional row position.

    This produces exact boundary coordinates even when the requested Y value
    lies between two measured data points.
    """

    lower_index = int(np.floor(position))

    if lower_index >= x_values.size - 1:
        return float(x_values[-1]), float(y_values[-1])

    fraction = position - lower_index

    x_value = x_values[lower_index] + fraction * (
        x_values[lower_index + 1] - x_values[lower_index]
    )

    y_value = y_values[lower_index] + fraction * (
        y_values[lower_index + 1] - y_values[lower_index]
    )

    return float(x_value), float(y_value)


def _select_y_branch(
    x_values,
    y_values,
    range_start,
    range_end,
    branch_choice,
):
    """
    Select one continuous data branch connecting two requested Y values.

    A nonmonotonic curve may pass through the same Y range multiple times.
    Each uninterrupted path between the two Y boundaries is treated as a
    separate candidate branch.

    The user can select either the first or last matching branch according to
    the row order in the uploaded data.
    """

    start_positions = _crossing_positions(
        y_values,
        range_start,
    )

    end_positions = _crossing_positions(
        y_values,
        range_end,
    )

    if not start_positions or not end_positions:
        raise SlopeCalculationError(
            "One or both selected Y values are outside the dataset."
        )

    lower_y = min(range_start, range_end)
    upper_y = max(range_start, range_end)

    y_tolerance = np.finfo(float).eps * 100 * max(
        1.0,
        abs(float(lower_y)),
        abs(float(upper_y)),
        float(np.max(np.abs(y_values))),
    )

    candidates = []

    # Test every possible connection between a start-Y crossing and an end-Y
    # crossing. A valid branch must remain inside the selected Y band for the
    # complete path between the two boundaries.
    for start_position in start_positions:
        for end_position in end_positions:
            lower_position = min(
                start_position,
                end_position,
            )

            upper_position = max(
                start_position,
                end_position,
            )

            if np.isclose(
                lower_position,
                upper_position,
                rtol=0.0,
                atol=1e-12,
            ):
                continue

            first_inside_index = int(
                np.floor(lower_position)
            ) + 1

            last_inside_index = int(
                np.ceil(upper_position)
            )

            interior_y = y_values[
                first_inside_index:last_inside_index
            ]

            if interior_y.size and (
                np.min(interior_y) < lower_y - y_tolerance
                or np.max(interior_y) > upper_y + y_tolerance
            ):
                # The path moved outside the requested Y band. Therefore these
                # two crossings belong to different curve branches.
                continue

            candidate = (
                lower_position,
                upper_position,
            )

            # Avoid recording the same branch more than once.
            if not any(
                np.allclose(
                    candidate,
                    existing,
                    rtol=0.0,
                    atol=1e-10,
                )
                for existing in candidates
            ):
                candidates.append(candidate)

    if not candidates:
        raise SlopeCalculationError(
            "No continuous data branch connects the two selected Y values."
        )

    # Candidate order follows the original row order of the uploaded data.
    candidates.sort(
        key=lambda candidate: (
            candidate[0],
            candidate[1],
        )
    )

    normalized_choice = str(
        branch_choice
    ).strip().lower()

    if normalized_choice.startswith("last"):
        lower_position, upper_position = candidates[-1]

    elif normalized_choice.startswith("first"):
        lower_position, upper_position = candidates[0]

    else:
        raise SlopeCalculationError(
            "Y-branch choice must be either First matching branch or "
            "Last matching branch."
        )

    # Calculate exact coordinates at the two boundaries of the selected branch.
    lower_x, lower_y_value = _interpolate_xy_at_position(
        x_values,
        y_values,
        lower_position,
    )

    upper_x, upper_y_value = _interpolate_xy_at_position(
        x_values,
        y_values,
        upper_position,
    )

    # Include every measured data point located between the two interpolated
    # boundary points.
    interior_indices = np.arange(
        int(np.floor(lower_position)) + 1,
        int(np.ceil(upper_position)),
        dtype=int,
    )

    branch_x = np.concatenate(
        (
            [lower_x],
            x_values[interior_indices],
            [upper_x],
        )
    )

    branch_y = np.concatenate(
        (
            [lower_y_value],
            y_values[interior_indices],
            [upper_y_value],
        )
    )

    return branch_x, branch_y


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
    """
    Clip a directed line segment to optional visible plot boundaries.

    The direction from the entered start to the entered end is retained. This
    ensures that the reported differences can correctly be positive or negative.
    """

    x_low, x_high = _ordered_plot_bounds(
        x_min,
        x_max,
    )

    y_low, y_high = _ordered_plot_bounds(
        y_min,
        y_max,
    )

    dx = x_end - x_start
    dy = y_end - y_start

    t_enter = 0.0
    t_exit = 1.0

    def constrain(
        value_start,
        difference,
        lower,
        upper,
        current_enter,
        current_exit,
    ):
        """Clip one coordinate of the parameterized line segment."""

        if difference == 0:
            if lower is not None and value_start < lower:
                return None

            if upper is not None and value_start > upper:
                return None

            return current_enter, current_exit

        if lower is not None:
            boundary_t = (
                lower - value_start
            ) / difference

            if difference > 0:
                current_enter = max(
                    current_enter,
                    boundary_t,
                )
            else:
                current_exit = min(
                    current_exit,
                    boundary_t,
                )

        if upper is not None:
            boundary_t = (
                upper - value_start
            ) / difference

            if difference > 0:
                current_exit = min(
                    current_exit,
                    boundary_t,
                )
            else:
                current_enter = max(
                    current_enter,
                    boundary_t,
                )

        if current_enter > current_exit:
            return None

        return current_enter, current_exit

    # First clip the segment against visible X boundaries.
    clipped = constrain(
        x_start,
        dx,
        x_low,
        x_high,
        t_enter,
        t_exit,
    )

    if clipped is None:
        raise SlopeCalculationError(
            "The requested slope does not intersect the visible X-axis range."
        )

    t_enter, t_exit = clipped

    # Then clip the remaining segment against visible Y boundaries.
    clipped = constrain(
        y_start,
        dy,
        y_low,
        y_high,
        t_enter,
        t_exit,
    )

    if clipped is None:
        raise SlopeCalculationError(
            "The requested slope does not intersect the visible Y-axis range."
        )

    t_enter, t_exit = clipped

    clipped_x_start = x_start + t_enter * dx
    clipped_y_start = y_start + t_enter * dy
    clipped_x_end = x_start + t_exit * dx
    clipped_y_end = y_start + t_exit * dy

    return (
        clipped_x_start,
        clipped_y_start,
        clipped_x_end,
        clipped_y_end,
    )


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
    y_branch="First matching branch",
):
    """
    Fit one straight segment inside an X or Y selection range.

    X-defined selection:
        Fits Y as a function of X.

    Y-defined selection:
        Isolates one continuous curve branch and fits X as a function of Y.
        This remains stable for vertical or nearly vertical branches.

    The resulting segment follows the entered start-to-end direction and is
    clipped against customized visible plot limits.
    """

    x_values, y_values = _finite_xy(
        x_values,
        y_values,
    )

    if x_values.size < 2:
        raise SlopeCalculationError(
            "At least two finite X-Y points are required."
        )

    if range_start == range_end:
        raise SlopeCalculationError(
            "The slope range start and end cannot be equal."
        )

    lower_range = min(
        range_start,
        range_end,
    )

    upper_range = max(
        range_start,
        range_end,
    )

    normalized_axis = range_axis.strip().lower()

    # --------------------------------------------------------
    # X-DEFINED RANGE
    # All points inside the numerical X interval are selected.
    # --------------------------------------------------------
    if normalized_axis.startswith("x"):
        selected = (
            (x_values >= lower_range)
            & (x_values <= upper_range)
        )

        selected_x = x_values[selected]
        selected_y = y_values[selected]

    # --------------------------------------------------------
    # Y-DEFINED RANGE
    # Only one continuous branch is selected. This prevents
    # separate loading and unloading sections from being mixed.
    # --------------------------------------------------------
    elif normalized_axis.startswith("y"):
        selected_x, selected_y = _select_y_branch(
            x_values,
            y_values,
            range_start,
            range_end,
            y_branch,
        )

    else:
        raise SlopeCalculationError(
            "Range axis must be either X axis or Y axis."
        )

    if selected_x.size < 2:
        raise SlopeCalculationError(
            "The selected range contains fewer than two data points."
        )

    # --------------------------------------------------------
    # FIT AN X-DEFINED SEGMENT
    # Standard least-squares regression: Y = slope*X + intercept
    # --------------------------------------------------------
    if normalized_axis.startswith("x"):

        if np.unique(selected_x).size < 2:
            raise SlopeCalculationError(
                "The selected range must contain at least two different X values."
            )

        slope, intercept = np.polyfit(
            selected_x,
            selected_y,
            1,
        )

        slope = float(slope)
        intercept = float(intercept)

        x_start = float(range_start)
        x_end = float(range_end)

        y_start = slope * x_start + intercept
        y_end = slope * x_end + intercept

    # --------------------------------------------------------
    # FIT A Y-DEFINED SEGMENT
    # Inverse regression: X = x_per_y*Y + x_intercept
    #
    # This avoids instability when X changes very little, such
    # as the nearly vertical section in the supplied example.
    # --------------------------------------------------------
    else:

        if np.unique(selected_y).size < 2:
            raise SlopeCalculationError(
                "The selected branch must contain at least two different Y values."
            )

        x_per_y, x_intercept = np.polyfit(
            selected_y,
            selected_x,
            1,
        )

        x_per_y = float(x_per_y)
        x_intercept = float(x_intercept)

        y_start = float(range_start)
        y_end = float(range_end)

        x_start = (
            x_per_y * y_start
            + x_intercept
        )

        x_end = (
            x_per_y * y_end
            + x_intercept
        )

        delta_x = x_end - x_start
        delta_y = y_end - y_start

        vertical_tolerance = np.finfo(float).eps * 100 * max(
            1.0,
            abs(x_start),
            abs(x_end),
        )

        if abs(delta_x) <= vertical_tolerance:

            # A vertical line has zero horizontal change. Its mathematical
            # slope ΔY/ΔX is infinite and its Y-intercept is not defined.
            slope = float("inf")
            intercept = float("nan")

        else:
            slope = float(
                delta_y / delta_x
            )

            intercept = float(
                y_start - slope * x_start
            )

    # Adjust the calculated segment to the currently visible plot boundaries.
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
    """
    Draw the calculated slope segment on a Matplotlib axis.

    display_y_values is used by the offset-stacked layout because its displayed
    Y positions are normalized while the reported slope coordinates remain in
    their original physical units.
    """

    if display_y_values is None:
        plotted_y = [
            slope_result.y_start,
            slope_result.y_end,
        ]
    else:
        plotted_y = display_y_values

    axis.plot(
        [
            slope_result.x_start,
            slope_result.x_end,
        ],
        plotted_y,
        label=label,
        color=color,
        linewidth=line_width,
        linestyle=line_style,
        zorder=10,
    )
