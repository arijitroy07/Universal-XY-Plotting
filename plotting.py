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


def apply_axis_settings(ax, x_min, x_max, y_min, y_max, x_tick, y_tick, grid):
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    if x_tick is not None and x_tick > 0:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick))
    if y_tick is not None and y_tick > 0:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick))
    if grid:
        ax.grid(True, alpha=0.25)


def normalize_trace(values):
    """Scale one trace to 0-1 using that trace's own finite min and max."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    minimum = np.min(finite)
    maximum = np.max(finite)
    if maximum == minimum:
        return np.full_like(values, 0.5)
    return (values - minimum) / (maximum - minimum)


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
