import io

import numpy as np
from matplotlib.ticker import MultipleLocator


def parse_optional_float(value):
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def add_dataset_to_axis(
    ax, x, y, label, plot_style, line_width, marker_size, color=None
):
    style = {"label": label}
    if color:
        style["color"] = color
    if plot_style == "Line":
        ax.plot(x, y, linewidth=line_width, **style)
    elif plot_style == "Scatter":
        ax.scatter(x, y, s=marker_size**2, **style)
    else:
        ax.plot(
            x,
            y,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            **style,
        )


def visible_x_mask(x_values, x_min=None, x_max=None):
    """
    Return a mask for data inside the selected X-axis range.

    X minimum may be greater than X maximum when the axis is reversed.
    """
    x_values = np.asarray(x_values, dtype=float)
    mask = np.isfinite(x_values)

    if x_min is not None and x_max is not None:
        lower_x = min(x_min, x_max)
        upper_x = max(x_min, x_max)

        mask &= (
            (x_values >= lower_x)
            & (x_values <= upper_x)
        )

    elif x_min is not None:
        mask &= x_values >= x_min

    elif x_max is not None:
        mask &= x_values <= x_max

    return mask


def visible_y_limits(
    xy_datasets,
    x_min=None,
    x_max=None,
    padding_fraction=0.05,
):
    """
    Calculate Y limits using only data visible inside the selected X range.

    xy_datasets must contain:
        [(x_array, y_array), ...]
    """
    visible_values = []

    for x_values, y_values in xy_datasets:
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)

        mask = visible_x_mask(
            x_values,
            x_min,
            x_max,
        )

        mask &= np.isfinite(y_values)

        if np.any(mask):
            visible_values.append(y_values[mask])

    if not visible_values:
        return None, None

    combined_y = np.concatenate(visible_values)

    minimum_y = float(np.min(combined_y))
    maximum_y = float(np.max(combined_y))

    y_range = maximum_y - minimum_y

    if y_range == 0:
        padding = max(
            abs(minimum_y) * padding_fraction,
            1.0 if minimum_y == 0 else 1e-9,
        )
    else:
        padding = y_range * padding_fraction

    return (
        minimum_y - padding,
        maximum_y + padding,
    )


def apply_axis_settings(
    ax,
    x_min,
    x_max,
    y_min,
    y_max,
    x_tick,
    y_tick,
    grid,
):
    """
    Apply axis settings.

    If an X range is entered and either Y limit is blank, the missing
    Y limit is calculated from only the data visible in the X range.
    """

    # Matplotlib accepts x_min > x_max and reverses the axis.
    if x_min is not None or x_max is not None:
        ax.set_xlim(
            left=x_min,
            right=x_max,
        )

    effective_y_min = y_min
    effective_y_max = y_max

    # Automatically fit Y to the visible X region.
    if (
        x_min is not None
        or x_max is not None
    ) and (
        y_min is None
        or y_max is None
    ):
        plotted_datasets = []

        # Line and scatter + line data
        for line in ax.get_lines():
            plotted_datasets.append(
                (
                    line.get_xdata(),
                    line.get_ydata(),
                )
            )

        # Scatter data
        for collection in ax.collections:
            offsets = collection.get_offsets()

            if offsets is not None and len(offsets):
                plotted_datasets.append(
                    (
                        offsets[:, 0],
                        offsets[:, 1],
                    )
                )

        auto_y_min, auto_y_max = visible_y_limits(
            plotted_datasets,
            x_min,
            x_max,
        )

        if effective_y_min is None:
            effective_y_min = auto_y_min

        if effective_y_max is None:
            effective_y_max = auto_y_max

    if (
        effective_y_min is not None
        or effective_y_max is not None
    ):
        ax.set_ylim(
            bottom=effective_y_min,
            top=effective_y_max,
        )

    if x_tick is not None and x_tick > 0:
        ax.xaxis.set_major_locator(
            MultipleLocator(x_tick)
        )

    if y_tick is not None and y_tick > 0:
        ax.yaxis.set_major_locator(
            MultipleLocator(y_tick)
        )

    if grid:
        ax.grid(True, alpha=0.25)


def normalize_trace(
    values,
    x_values=None,
    x_min=None,
    x_max=None,
):
    """
    Scale one trace from 0 to 1.

    When an X range is specified, the normalization minimum and maximum
    are calculated from only the visible portion of that trace.
    """
    values = np.asarray(values, dtype=float)
    reference_values = values

    if (
        x_values is not None
        and (
            x_min is not None
            or x_max is not None
        )
    ):
        mask = visible_x_mask(
            x_values,
            x_min,
            x_max,
        )

        mask &= np.isfinite(values)

        if np.any(mask):
            reference_values = values[mask]

    finite_reference = reference_values[
        np.isfinite(reference_values)
    ]

    if finite_reference.size == 0:
        return np.zeros_like(values)

    minimum = np.min(finite_reference)
    maximum = np.max(finite_reference)

    if maximum == minimum:
        return np.full_like(values, 0.5)

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def figure_bytes(fig, file_format, dpi=300):
    buffer = io.BytesIO()
    save_options = {"format": file_format, "bbox_inches": "tight"}
    if file_format in {"png", "jpg", "jpeg"}:
        save_options["dpi"] = dpi
    if file_format in {"jpg", "jpeg"}:
        save_options["pil_kwargs"] = {"quality": 95}
    fig.savefig(buffer, **save_options)
    buffer.seek(0)
    return buffer.getvalue()
