# ============================================================
# UNIVERSAL XY DATA PLOTTER - MAIN STREAMLIT APPLICATION
# ============================================================
#
# PURPOSE
# -------
# This file controls the complete user interface and coordinates the other
# project modules. It deliberately keeps file parsing, general plotting
# primitives, and slope mathematics in separate files so future changes can
# be made without rewriting the entire application.
#
# MODULE RESPONSIBILITIES
# -----------------------
# data_reader.py:
#     Opens the uploaded file, detects its table/header/units, and converts
#     the selected columns into usable numeric values.
# plotting.py:
#     Draws a dataset, applies axis/tick/grid settings, converts figures to
#     downloadable JPEG bytes, and parses optional numeric text fields.
# slope_tools.py:
#     Fits the optional straight slope line, clips it to visible axes, and
#     returns the physical start/end coordinates and signed differences.
# app.py (this file):
#     Builds the interface, stores user choices, selects the correct plotting
#     mode, coordinates the helper modules, and displays the final output.
#
# IMPORTANT STREAMLIT BEHAVIOR
# ----------------------------
# Streamlit reruns this entire file whenever any widget changes. The actual
# plot is therefore stored in st.session_state and is regenerated only when
# the user presses "Create / Update Plot". This prevents every small setting
# change from immediately redrawing the figure.

# Standard-library path handling, used for file extensions and default labels.
from pathlib import Path

# Matplotlib is forced to use a noninteractive backend because Streamlit Cloud
# runs without a desktop display. This line must appear before pyplot import.
import matplotlib
matplotlib.use("Agg")

# Third-party numerical, plotting, and web-interface packages.
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# File-reading functions. Keep instrument-export detection changes in
# data_reader.py rather than adding file-format logic directly to this file.
from data_reader import (
    clean_numeric_series,
    get_excel_sheets,
    prepare_table,
    read_uploaded_file,
)

# General Matplotlib functions shared by every plotting mode.
from plotting import (
    add_dataset_to_axis,
    apply_axis_settings,
    figure_bytes,
    parse_optional_float,
)

# Optional slope feature. All fitting and geometric clipping mathematics are
# intentionally isolated in slope_tools.py.
from slope_tools import (
    SlopeCalculationError,
    add_slope_line,
    calculate_slope_segment,
)


# ============================================================
# PAGE-WIDE STREAMLIT SETTINGS
# This must be called before other Streamlit display commands.
# ============================================================

st.set_page_config(page_title="Universal XY Plotter", page_icon="📈", layout="wide")
st.title("Universal XY Data Plotter")
st.caption(
    "Upload XLS, XLSX, CSV, TXT, or XY files. Select X and Y columns for "
    "each dataset, then customize and export the plot."
)


# ============================================================
# OPTIONAL NUMERIC INPUT HELPER
# Axis fields are text inputs so the user can leave them completely blank.
# A blank field returns None, which tells Matplotlib to use automatic values.
# ============================================================

def optional_number(label: str, key: str):
    """A blank text field that returns either a float or None."""
    raw_value = st.text_input(label, value="", key=key)
    value = parse_optional_float(raw_value)

    if raw_value.strip() and value is None:
        st.warning(f"{label} must be a number; automatic scaling will be used.")

    return value


# ============================================================
# VISIBLE-RANGE HELPERS
# These preserve the prior reversed-X and visible-Y behavior.
# ============================================================

def visible_x_mask(x_values, x_min=None, x_max=None):
    """Select points inside normal or reversed X-axis limits."""

    # Convert to a predictable floating-point NumPy array before filtering.
    x_values = np.asarray(x_values, dtype=float)
    mask = np.isfinite(x_values)

    # min()/max() are used only for data selection. The original user-entered
    # order is still passed to Matplotlib, so X minimum > X maximum continues
    # to display a reversed axis while the correct data window is selected.
    if x_min is not None and x_max is not None:
        lower_x = min(x_min, x_max)
        upper_x = max(x_min, x_max)
        mask &= (x_values >= lower_x) & (x_values <= upper_x)
    elif x_min is not None:
        mask &= x_values >= x_min
    elif x_max is not None:
        mask &= x_values <= x_max

    return mask


def visible_y_limits(xy_datasets, x_min=None, x_max=None, padding_fraction=0.05):
    """Calculate Y limits from data visible inside the selected X range."""

    # Multiple overlapping datasets are collected before a shared Y range is
    # calculated. For an individual panel, this list contains one dataset.
    visible_values = []

    for x_values, y_values in xy_datasets:
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        mask = visible_x_mask(x_values, x_min, x_max) & np.isfinite(y_values)

        if np.any(mask):
            visible_values.append(y_values[mask])

    if not visible_values:
        # Returning None leaves Matplotlib in automatic mode when the selected
        # X window does not contain any usable data.
        return None, None

    combined_y = np.concatenate(visible_values)
    minimum_y = float(np.min(combined_y))
    maximum_y = float(np.max(combined_y))
    y_range = maximum_y - minimum_y

    if y_range == 0:
        # A constant visible Y value needs nonzero padding or Matplotlib would
        # receive identical lower and upper limits.
        padding = (
            1.0
            if minimum_y == 0
            else max(abs(minimum_y) * padding_fraction, 1e-9)
        )
    else:
        padding = y_range * padding_fraction

    return minimum_y - padding, maximum_y + padding


def resolve_y_limits(
    xy_datasets,
    x_min,
    x_max,
    manual_y_min,
    manual_y_max,
):
    """Fill blank Y limits using only data visible in a customized X range."""

    # Manual Y values always take priority. Only a missing boundary is filled
    # from the data visible inside a customized X window.
    effective_y_min = manual_y_min
    effective_y_max = manual_y_max

    if x_min is None and x_max is None:
        # With no custom X window, normal Matplotlib autoscaling is retained.
        return effective_y_min, effective_y_max

    auto_y_min, auto_y_max = visible_y_limits(xy_datasets, x_min, x_max)

    if effective_y_min is None:
        effective_y_min = auto_y_min

    if effective_y_max is None:
        effective_y_max = auto_y_max

    return effective_y_min, effective_y_max


def visible_trace_y_bounds(y_values, x_values, x_min=None, x_max=None):
    """Return one trace's Y min/max inside the visible X range."""

    # Offset-stacked plots normalize every trace independently. These bounds
    # therefore belong to one dataset, not to the combined collection.
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = visible_x_mask(x_values, x_min, x_max) & np.isfinite(y_values)

    reference = (
        y_values[mask]
        if np.any(mask)
        else y_values[np.isfinite(y_values)]
    )

    if reference.size == 0:
        # Defensive fallback for an empty or fully nonnumeric trace.
        return 0.0, 0.0

    return float(np.min(reference)), float(np.max(reference))


def scale_values(values, minimum, maximum):
    """Scale values to 0-1 using supplied reference bounds."""

    # This normalized coordinate is used only for visual separation in the
    # offset-stacked mode. Slope text output remains in original data units.
    values = np.asarray(values, dtype=float)

    if maximum == minimum:
        return np.full_like(values, 0.5)

    return (values - minimum) / (maximum - minimum)


def format_output_number(value):
    """Compact numeric formatting for slope coordinate output."""

    # Eight significant digits avoids unnecessary trailing zeros while still
    # retaining useful precision for scientific datasets.
    return f"{value:.8g}"


# ============================================================
# SECTION 1 - FILE UPLOAD
# Uploading does not create a plot. It only makes the files available for
# detection and column selection below.
# ============================================================

st.header("1. Upload data")

uploaded_files = st.file_uploader(
    "Upload one or more data files",
    type=["xls", "xlsx", "csv", "txt", "xy"],
    accept_multiple_files=True,
)

if not uploaded_files:
    # st.stop() prevents the remaining widgets from running before data exist.
    st.info("Upload at least one data file to continue.")
    st.stop()


# ============================================================
# SECTION 2 - PER-FILE TABLE, COLUMN, LABEL, AND COLOR SETTINGS
# Each uploaded file is processed independently and becomes one dataset.
# ============================================================

st.header("2. Select X and Y data")

datasets = []

for file_index, uploaded_file in enumerate(uploaded_files):

    # Streamlit widget keys must be unique. Including the file index and name
    # prevents controls for one uploaded file from changing another file.
    file_key = f"file_{file_index}_{uploaded_file.name}"

    with st.expander(f"{file_index + 1}. {uploaded_file.name}", expanded=True):
        extension = Path(uploaded_file.name).suffix.lower()
        selected_sheet = None

        try:
            # Excel workbooks require a sheet choice; text formats do not.
            if extension in {".xls", ".xlsx"}:
                sheets = get_excel_sheets(uploaded_file.getvalue())

                selected_sheet = st.selectbox(
                    "Excel sheet",
                    sheets,
                    key=f"sheet_{file_key}",
                )

            raw_df, detected_separator = read_uploaded_file(
                uploaded_file,
                selected_sheet,
            )

        except Exception as error:
            st.error(f"Could not read this file: {error}")
            continue

        with st.expander("Detection settings and raw preview"):

            # The raw preview helps diagnose unusual instrument export layouts
            # before automatic or manual header selection is applied.
            st.dataframe(raw_df.head(20), width="stretch")

            detection_mode = st.radio(
                "Table detection",
                ["Automatic", "Manual override"],
                horizontal=True,
                key=f"detection_{file_key}",
            )

            data_start_override = None
            header_row_override = None
            unit_row_override = None

            if detection_mode == "Manual override":

                # User-facing rows are 1-based. The reader internally uses
                # zero-based DataFrame indices, so every nonzero choice is
                # reduced by one before being passed to prepare_table().
                st.caption(
                    "Rows are numbered from 1. Use 0 for no header or units row."
                )

                row_col1, row_col2, row_col3 = st.columns(3)

                with row_col1:
                    data_start_override = int(
                        st.number_input(
                            "First numeric data row",
                            min_value=1,
                            max_value=max(1, len(raw_df)),
                            value=min(2, max(1, len(raw_df))),
                            step=1,
                            key=f"data_start_{file_key}",
                        )
                    ) - 1

                with row_col2:
                    header_choice = int(
                        st.number_input(
                            "Header row (0 = none)",
                            min_value=0,
                            max_value=max(0, len(raw_df)),
                            value=1 if len(raw_df) else 0,
                            step=1,
                            key=f"header_row_{file_key}",
                        )
                    )

                    header_row_override = (
                        header_choice - 1
                        if header_choice
                        else None
                    )

                with row_col3:
                    unit_choice = int(
                        st.number_input(
                            "Units row (0 = none)",
                            min_value=0,
                            max_value=max(0, len(raw_df)),
                            value=0,
                            step=1,
                            key=f"unit_row_{file_key}",
                        )
                    )

                    unit_row_override = (
                        unit_choice - 1
                        if unit_choice
                        else None
                    )

        try:
            # prepare_table() identifies active numeric columns and combines a
            # separate unit row with the detected column names when available.
            prepared_df, metadata = prepare_table(
                raw_df,
                data_start_override=data_start_override,
                header_row_override=header_row_override,
                unit_row_override=unit_row_override,
            )

        except Exception as error:
            st.error(f"Could not detect a usable table: {error}")
            continue

        if prepared_df.shape[1] < 2:
            st.warning(
                "At least two usable numeric columns could not be detected."
            )
            continue

        detected_columns = list(prepared_df.columns)

        # Detection details are shown so the user can confirm that the correct
        # rows were interpreted as data, headers, and units.
        detection_text = (
            f"Numeric data begins near row {metadata['data_start'] + 1}."
        )

        if metadata["header_row"] is not None:
            detection_text += (
                f" Header row: {metadata['header_row'] + 1}."
            )

        if metadata["unit_row"] is not None:
            detection_text += (
                f" Units row: {metadata['unit_row'] + 1}."
            )

        if detected_separator:
            detection_text += (
                f" Detected separator: {detected_separator!r}."
            )

        st.caption(detection_text)

        st.write("**Cleaned preview**")
        st.dataframe(prepared_df.head(10), width="stretch")

        selection_col1, selection_col2 = st.columns(2)

        # X and Y choices are intentionally independent for every uploaded file.
        with selection_col1:
            x_column = st.selectbox(
                "X-axis column",
                detected_columns,
                index=0,
                key=f"x_{file_key}",
            )

        with selection_col2:
            y_column = st.selectbox(
                "Y-axis column",
                detected_columns,
                index=min(1, len(detected_columns) - 1),
                key=f"y_{file_key}",
            )

        label_col, color_col = st.columns([2, 1])

        # The filename stem is only a default. The editable label is what later
        # appears in the plot legend and in the slope-dataset selector.
        with label_col:
            legend_label = st.text_input(
                "Legend label",
                value=Path(uploaded_file.name).stem,
                key=f"legend_{file_key}",
            )

        with color_col:
            use_custom_color = st.checkbox(
                "Custom color",
                value=False,
                key=f"use_color_{file_key}",
            )

            color = (
                st.color_picker(
                    "Trace color",
                    "#1f77b4",
                    key=f"color_{file_key}",
                )
                if use_custom_color
                else None
            )

        x_numeric = clean_numeric_series(prepared_df[x_column])
        y_numeric = clean_numeric_series(prepared_df[y_column])

        # A point is retained only when both its X and Y values are numeric.
        # The original row order is preserved for line plots.
        valid = x_numeric.notna() & y_numeric.notna()
        x_numeric = x_numeric[valid]
        y_numeric = y_numeric[valid]

        if x_numeric.empty:
            st.warning(
                "No valid numeric X-Y pairs were found for these columns."
            )
            continue

        st.success(f"{len(x_numeric):,} valid X-Y points detected.")

        datasets.append(
            # This dictionary is the common internal representation consumed
            # by every layout mode and by the optional slope feature.
            {
                "file": uploaded_file.name,
                "label": legend_label,
                "x": x_numeric.to_numpy(),
                "y": y_numeric.to_numpy(),
                "x_column": x_column,
                "y_column": y_column,
                "color": color,
            }
        )

if not datasets:
    # Files may have uploaded successfully but still contain no selectable
    # numeric X-Y pairs. Stop before accessing datasets[0] later in the UI.
    st.error("No usable datasets are available.")
    st.stop()


# ============================================================
# SECTION 3 - GENERAL PLOT CONFIGURATION
# These controls are shared by all plotting modes.
# ============================================================

st.header("3. Configure plot")

layout_mode = st.radio(
    "Multiple-file plotting mode",
    ["Overlap", "Stacked traces with Y offset", "Stacked panels"],
    horizontal=True,
)

plot_style = st.radio(
    "Plot type",
    ["Line", "Scatter", "Scatter + Line"],
    horizontal=True,
)


# ============================================================
# FIGURE SIZE AND TRACE APPEARANCE
# Physical figure dimensions affect the aspect ratio shown on screen and the
# dimensions of the downloaded JPEG. Line width and marker size are applied
# to every data trace.
# ============================================================

st.subheader("Figure and trace style")

size_col1, size_col2, style_col1, style_col2 = st.columns(4)

with size_col1:
    figure_width = st.number_input(
        "Width (inches)",
        min_value=2.0,
        max_value=30.0,
        value=8.0,
        step=0.5,
    )

with size_col2:
    figure_height = st.number_input(
        "Height (inches)",
        min_value=2.0,
        max_value=40.0,
        value=6.0,
        step=0.5,
    )

with style_col1:
    line_width = st.number_input(
        "Line width",
        min_value=0.1,
        max_value=10.0,
        value=1.5,
        step=0.1,
    )

with style_col2:
    marker_size = st.number_input(
        "Marker size",
        min_value=1.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
    )

st.caption(
    f"Current aspect ratio: {figure_width / figure_height:.2f}:1"
)


# ============================================================
# TITLES, AXIS LABELS, AND LEGEND TITLE
# These fields change only displayed text. Matplotlib math text can be used,
# for example MoS$_2$ or g$^{-1}$.
# ============================================================

st.subheader("Titles and labels")

plot_title = st.text_input("Plot title", value="")

label_col1, label_col2 = st.columns(2)

with label_col1:
    x_axis_label = st.text_input(
        "X-axis label",
        value=datasets[0]["x_column"],
    )

with label_col2:
    y_axis_label = st.text_input(
        "Y-axis label",
        value=datasets[0]["y_column"],
    )

legend_title = st.text_input("Legend title", value="")


# ============================================================
# AXIS LIMITS AND TICK INTERVALS
# Blank fields mean automatic scaling. X minimum may be greater than X maximum
# to intentionally reverse the X-axis.
# ============================================================

st.subheader("Axis limits and tick spacing")
st.caption("Leave a field blank to use automatic values.")

if layout_mode == "Stacked traces with Y offset":

    # Offset-stacked traces do not expose physical Y coordinates on the plot,
    # so only X limits and X ticks are configurable in this layout.
    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:
        x_min = optional_number("X minimum", "x_min")

    with axis_col2:
        x_max = optional_number("X maximum", "x_max")

    with axis_col3:
        x_tick = optional_number(
            "X major tick interval",
            "x_tick",
        )

    y_min = None
    y_max = None
    y_tick = None

    st.info(
        "For offset-stacked traces, every dataset is automatically scaled "
        "using its own Y minimum and maximum. Y-axis numbers are hidden."
    )

else:

    # Overlap and stacked-panel layouts retain normal physical Y coordinates.
    axis_cols = st.columns(4)

    with axis_cols[0]:
        x_min = optional_number("X minimum", "x_min")

    with axis_cols[1]:
        x_max = optional_number("X maximum", "x_max")

    with axis_cols[2]:
        y_min = optional_number("Y minimum", "y_min")

    with axis_cols[3]:
        y_max = optional_number("Y maximum", "y_max")

    tick_col1, tick_col2 = st.columns(2)

    with tick_col1:
        x_tick = optional_number(
            "X major tick interval",
            "x_tick",
        )

    with tick_col2:
        y_tick = optional_number(
            "Y major tick interval",
            "y_tick",
        )

st.caption(
    "To reverse the X-axis, enter an X minimum greater than the X maximum. "
    "When Y limits are blank, they automatically fit the visible data."
)

if x_min is not None and x_max is not None and x_min == x_max:

    # Reversed X limits are valid; only identical X limits are impossible.
    st.error("X minimum and X maximum cannot be equal.")
    st.stop()

if y_min is not None and y_max is not None and y_min >= y_max:

    # Y reversal was not requested, so Y minimum must remain below Y maximum.
    st.error("Y minimum must be smaller than Y maximum.")
    st.stop()


# ============================================================
# LEGEND AND GRID DISPLAY OPTIONS
# These controls alter the plot appearance but not the underlying data.
# ============================================================

option_col1, option_col2, option_col3 = st.columns(3)

with option_col1:
    show_legend = st.checkbox(
        "Show legend",
        value=True,
    )

with option_col2:
    show_grid = st.checkbox(
        "Show grid",
        value=False,
    )

with option_col3:
    legend_location = st.selectbox(
        "Legend position",
        [
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "center right",
            "center left",
            "upper center",
            "lower center",
        ],
    )


# ============================================================
# LAYOUT-SPECIFIC SETTINGS
# Only the control relevant to the selected plot arrangement is displayed.
# ============================================================

fixed_panel_y = False

if layout_mode == "Stacked traces with Y offset":

    # A normalized trace occupies one vertical unit. trace_gap adds empty
    # vertical space between consecutive normalized traces.
    trace_gap = st.slider(
        "Space between stacked traces",
        min_value=0.0,
        max_value=2.0,
        value=0.25,
        step=0.05,
    )

elif layout_mode == "Stacked panels":

    # When disabled, each panel receives its own automatic visible-data Y range.
    # When enabled, all panels use one combined visible-data Y range.
    fixed_panel_y = st.checkbox(
        "Use one shared Y-axis range for all panels",
        value=False,
    )


# ============================================================
# OPTIONAL SLOPE FEATURE
# Nothing in this section affects the plot unless the checkbox is selected.
#
# WORKFLOW
# --------
# 1. Choose one dataset when several files are uploaded.
# 2. Choose whether fitting points are selected using an X range or Y range.
# 3. Enter the directed start and end values. Their order is retained, allowing
#    the final coordinate differences to be positive or negative.
# 4. slope_tools.py performs a least-squares fit of Y = mX + b.
# 5. The fitted segment is clipped against customized visible plot limits.
# 6. Its visible start, end, and signed differences are displayed as text.
# ============================================================

st.subheader("Optional slope line")

show_slope = st.checkbox(
    "Draw a fitted slope line",
    value=False,
)

# Defaults are defined even when the checkbox is not selected. This ensures
# these variables always exist when the plotting section is evaluated.
slope_dataset_index = 0
slope_range_axis = "X axis"
slope_range_start = None
slope_range_end = None
slope_line_label = "Slope fit"
slope_line_color = "#D62728"
slope_line_width = 2.0
slope_line_style = "--"

# Determines which continuous curve branch is used when the selected
# Y range occurs more than once in the dataset.
slope_y_branch = "First matching branch"

if show_slope:

    # Slope controls are created only after the feature is selected. Keeping
    # them conditional prevents additional settings from cluttering normal use.
    st.caption(
        "The app fits Y = mX + b using points inside the selected X or Y "
        "range. For a repeated Y range, choose which continuous curve branch "
        "to use. The fitted segment is clipped to customized plot limits."
    )

    slope_dataset_index = st.selectbox(

        # The index is stored rather than the label because labels may be
        # duplicated or edited by the user.
        "Dataset used for the slope",
        options=list(range(len(datasets))),
        format_func=lambda index: (
            f"{datasets[index]['label']} ({datasets[index]['file']})"
        ),
    )

    slope_range_axis = st.radio(
        "Define the fitting range using",
        ["X axis", "Y axis"],
        horizontal=True,
    )

    if slope_range_axis == "Y axis":
    slope_y_branch = st.selectbox(
        "When this Y range occurs more than once",
        [
            "First matching branch",
            "Last matching branch",
        ],
        help=(
            "First matching branch uses the earliest matching section in "
            "the uploaded row order. Last matching branch uses the latest "
            "matching section."
        ),
    )


    
    slope_dataset = datasets[slope_dataset_index]

    # Default range endpoints use the middle 50% of the chosen coordinate.
    # This normally provides enough points for an initial fit while avoiding
    # automatic use of the extreme edges of the dataset.
    range_values = (
        slope_dataset["x"]
        if slope_range_axis == "X axis"
        else slope_dataset["y"]
    )

    finite_range_values = np.asarray(
        range_values,
        dtype=float,
    )

    finite_range_values = finite_range_values[
        np.isfinite(finite_range_values)
    ]

    if finite_range_values.size:
        default_start, default_end = np.percentile(
            finite_range_values,
            [25, 75],
        )
    else:
        default_start, default_end = 0.0, 1.0

    range_col1, range_col2 = st.columns(2)

    # Separate widget keys preserve independent remembered values when the user
    # switches between X-defined and Y-defined ranges.
    range_key = (
        "x"
        if slope_range_axis == "X axis"
        else "y"
    )

    with range_col1:
        slope_range_start = st.number_input(
            f"{slope_range_axis} range start",
            value=float(default_start),
            format="%.8f",
            key=f"slope_range_start_{range_key}",
        )

    with range_col2:
        slope_range_end = st.number_input(
            f"{slope_range_axis} range end",
            value=float(default_end),
            format="%.8f",
            key=f"slope_range_end_{range_key}",
        )

    slope_style_col1, slope_style_col2, slope_style_col3 = st.columns(3)

    # These settings affect only the appearance and legend entry of the fitted
    # line. They do not change the fitted coordinates or calculated slope.
    with slope_style_col1:
        slope_line_label = st.text_input(
            "Slope legend label",
            value="Slope fit",
        )

    with slope_style_col2:
        slope_line_color = st.color_picker(
            "Slope line color",
            "#D62728",
        )

    with slope_style_col3:
        slope_line_width = st.number_input(
            "Slope line width",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
        )

    slope_line_style = st.selectbox(
        "Slope line style",
        ["--", "-", ":", "-."],
        format_func=lambda style: {
            "--": "Dashed",
            "-": "Solid",
            ":": "Dotted",
            "-.": "Dash-dot",
        }[style],
    )


# ============================================================
# SECTION 4 - EXPLICIT PLOT GENERATION
# No figure is generated until this button is pressed. After generation, the
# JPEG bytes and optional slope output are kept in Streamlit session state so
# they remain visible while the user reviews settings.
# ============================================================

st.header("4. Plot")

st.caption(
    "Changing a setting does not redraw the figure. Select the button below "
    "when you are ready to create or update it."
)

create_plot = st.button(
    "Create / Update Plot",
    type="primary",
    width="stretch",
)

if create_plot:

    # Clear previous output first. This prevents an old plot or old slope values
    # from appearing beneath an error produced by a new set of settings.
    st.session_state.pop("plot_jpeg", None)
    st.session_state.pop("slope_output", None)

    fig = None
    slope_result = None

    try:

        # ========================================================
        # MODE 1 - OVERLAP
        # Every dataset is drawn on one physical X-Y axis.
        # ========================================================
        if layout_mode == "Overlap":

            fig, ax = plt.subplots(
                figsize=(figure_width, figure_height)
            )

            for dataset in datasets:
                add_dataset_to_axis(
                    ax,
                    dataset["x"],
                    dataset["y"],
                    dataset["label"],
                    plot_style,
                    line_width,
                    marker_size,
                    dataset["color"],
                )

            # When X is customized and Y is blank, all visible overlapping
            # datasets contribute to one shared automatic Y range.
            effective_y_min, effective_y_max = resolve_y_limits(
                [
                    (dataset["x"], dataset["y"])
                    for dataset in datasets
                ],
                x_min,
                x_max,
                y_min,
                y_max,
            )

            if show_slope:

                # Fit in original data coordinates, then clip the fitted segment
                # to the same effective limits used for the displayed axis.
                slope_dataset = datasets[slope_dataset_index]

                slope_result = calculate_slope_segment(
                    slope_dataset["x"],
                    slope_dataset["y"],
                    slope_range_axis,
                    slope_range_start,
                    slope_range_end,
                    visible_x_min=x_min,
                    visible_x_max=x_max,
                    visible_y_min=effective_y_min,
                    visible_y_max=effective_y_max,
                    y_branch=slope_y_branch,
                )

                add_slope_line(
                    ax,
                    slope_result,
                    label=slope_line_label,
                    color=slope_line_color,
                    line_width=slope_line_width,
                    line_style=slope_line_style,
                )

            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)

            if plot_title:
                ax.set_title(plot_title)

            # apply_axis_settings() also preserves reversed X limits because
            # the original x_min/x_max order is passed unchanged.
            apply_axis_settings(
                ax,
                x_min,
                x_max,
                effective_y_min,
                effective_y_max,
                x_tick,
                y_tick,
                show_grid,
            )

            if show_legend:
                ax.legend(
                    title=legend_title or None,
                    loc=legend_location,
                )

            fig.tight_layout()


        # ========================================================
        # MODE 2 - STACKED TRACES WITH Y OFFSET
        # Each dataset is independently normalized to a 0-1 vertical band and
        # then shifted upward. Original Y units are preserved only in the slope
        # text output, not in the plotted vertical coordinate.
        # ========================================================
        elif layout_mode == "Stacked traces with Y offset":

            fig, ax = plt.subplots(
                figsize=(figure_width, figure_height)
            )

            # Each normalized trace has a height of approximately 1.0.
            # trace_gap increases the separation between those trace bands.
            trace_step = 1.0 + trace_gap

            for dataset_index, dataset in enumerate(datasets):

                # The normalization bounds are recalculated from only the data
                # visible inside a customized X range.
                reference_y_min, reference_y_max = visible_trace_y_bounds(
                    dataset["y"],
                    dataset["x"],
                    x_min,
                    x_max,
                )

                independently_scaled_y = scale_values(
                    dataset["y"],
                    reference_y_min,
                    reference_y_max,
                )

                # Move the normalized trace into its individual vertical band.
                stacked_y = (
                    independently_scaled_y
                    + dataset_index * trace_step
                )

                add_dataset_to_axis(
                    ax,
                    dataset["x"],
                    stacked_y,
                    dataset["label"],
                    plot_style,
                    line_width,
                    marker_size,
                    dataset["color"],
                )

                if show_slope and dataset_index == slope_dataset_index:

                    # Fit and report the slope in original physical coordinates.
                    slope_result = calculate_slope_segment(
                        dataset["x"],
                        dataset["y"],
                        slope_range_axis,
                        slope_range_start,
                        slope_range_end,
                        visible_x_min=x_min,
                        visible_x_max=x_max,
                        y_branch=slope_y_branch,
                    )

                    # Only the two displayed slope Y coordinates are mapped into
                    # the selected trace's normalized offset band.
                    displayed_slope_y = scale_values(
                        [
                            slope_result.y_start,
                            slope_result.y_end,
                        ],
                        reference_y_min,
                        reference_y_max,
                    ) + dataset_index * trace_step

                    add_slope_line(
                        ax,
                        slope_result,
                        label=slope_line_label,
                        color=slope_line_color,
                        line_width=slope_line_width,
                        line_style=slope_line_style,
                        display_y_values=displayed_slope_y,
                    )

            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(y_axis_label)

            if plot_title:
                ax.set_title(plot_title)

            # Y limits and Y tick intervals are not passed here because the
            # displayed Y positions are normalized trace offsets.
            apply_axis_settings(
                ax,
                x_min,
                x_max,
                None,
                None,
                x_tick,
                None,
                show_grid,
            )

            # Add a small margin below the first trace and above the final trace.
            ax.set_ylim(
                -0.05,
                (len(datasets) - 1) * trace_step + 1.05,
            )

            # Numerical Y ticks are intentionally removed in this mode. The
            # user-entered Y-axis label remains visible.
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                labelleft=False,
            )

            ax.set_yticks([])

            if show_legend:
                ax.legend(
                    title=legend_title or None,
                    loc=legend_location,
                )

            fig.tight_layout()


        # ========================================================
        # MODE 3 - STACKED PANELS
        # Every dataset receives a separate vertical subplot with a shared X
        # axis. A slope is drawn only on the panel belonging to its dataset.
        # ========================================================
        else:

            number_of_plots = len(datasets)

            fig, axes = plt.subplots(
                number_of_plots,
                1,
                figsize=(figure_width, figure_height),
                sharex=True,
                squeeze=False,
            )

            # squeeze=False guarantees a 2D axes array. Ravel converts it into
            # a simple one-dimensional sequence for the plotting loop.
            axes = axes.ravel()

            common_y_min = None
            common_y_max = None

            if fixed_panel_y:

                # One visible-data Y range is calculated from all datasets and
                # reused for every panel. Manual Y boundaries still take priority.
                auto_common_y_min, auto_common_y_max = visible_y_limits(
                    [
                        (dataset["x"], dataset["y"])
                        for dataset in datasets
                    ],
                    x_min,
                    x_max,
                )

                common_y_min = (
                    y_min
                    if y_min is not None
                    else auto_common_y_min
                )

                common_y_max = (
                    y_max
                    if y_max is not None
                    else auto_common_y_max
                )

            for dataset_index, (ax, dataset) in enumerate(
                zip(axes, datasets)
            ):
                add_dataset_to_axis(
                    ax,
                    dataset["x"],
                    dataset["y"],
                    dataset["label"],
                    plot_style,
                    line_width,
                    marker_size,
                    dataset["color"],
                )

                if fixed_panel_y:

                    # Shared limits support direct magnitude comparison across
                    # panels, although small variations may become less visible.
                    panel_y_min = common_y_min
                    panel_y_max = common_y_max

                else:

                    # Independent limits maximize the visibility of each panel.
                    panel_y_min, panel_y_max = resolve_y_limits(
                        [(dataset["x"], dataset["y"])],
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                    )

                if show_slope and dataset_index == slope_dataset_index:

                    # Only the selected dataset's subplot receives the fit line.
                    slope_result = calculate_slope_segment(
                        dataset["x"],
                        dataset["y"],
                        slope_range_axis,
                        slope_range_start,
                        slope_range_end,
                        visible_x_min=x_min,
                        visible_x_max=x_max,
                        visible_y_min=panel_y_min,
                        visible_y_max=panel_y_max,
                        y_branch=slope_y_branch,
                    )

                    add_slope_line(
                        ax,
                        slope_result,
                        label=slope_line_label,
                        color=slope_line_color,
                        line_width=slope_line_width,
                        line_style=slope_line_style,
                    )

                apply_axis_settings(
                    ax,
                    x_min,
                    x_max,
                    panel_y_min,
                    panel_y_max,
                    x_tick,
                    y_tick,
                    show_grid,
                )

                if show_legend:
                    ax.legend(
                        title=legend_title or None,
                        loc=legend_location,
                    )

            # Only the bottom subplot receives the X label because all panels
            # share the same X-axis.
            axes[-1].set_xlabel(x_axis_label)

            # A single shared Y label avoids repeating identical text beside
            # every vertically stacked panel.
            fig.supylabel(y_axis_label)

            if plot_title:
                fig.suptitle(plot_title)

                # Reserve space at the top for the figure-level title.
                fig.tight_layout(rect=[0, 0, 1, 0.96])
            else:
                fig.tight_layout()


        # ========================================================
        # SAVE THE COMPLETED FIGURE IN SESSION STATE
        # The rendered figure is converted once and stored as JPEG bytes.
        # Widget changes after this point do not redraw it until the user
        # presses the plot button again.
        # ========================================================

        st.session_state["plot_jpeg"] = figure_bytes(
            fig,
            "jpeg",
            dpi=300,
        )

        if slope_result is not None:

            # Dataclasses are converted to a plain dictionary because simple
            # serializable values are safest for Streamlit session state.
            st.session_state["slope_output"] = (
                slope_result.to_dict()
            )


    # ========================================================
    # ERROR HANDLING
    # SlopeCalculationError covers expected slope-selection problems.
    # Other exceptions are displayed as general plot-creation errors.
    # ========================================================

    except SlopeCalculationError as error:

        # Expected user-correctable cases include too few selected points, a
        # zero fitted slope for a Y-defined range, or no visible intersection.
        st.error(
            f"The slope could not be created: {error}"
        )

    except Exception as error:

        # Other reading, plotting, or rendering problems use a general message.
        st.error(
            f"The plot could not be created: {error}"
        )

    finally:

        # Matplotlib figures must be closed to avoid accumulating memory across
        # repeated Streamlit reruns and plot updates.
        if fig is not None:
            plt.close(fig)


# ============================================================
# PERSISTED PLOT DISPLAY, SLOPE TEXT OUTPUT, AND JPEG DOWNLOAD
# This section reads the last successfully generated result from Streamlit
# session state. It does not calculate or redraw the plot.
# ============================================================

if "plot_jpeg" in st.session_state:

    # Display the exact JPEG data that will also be offered for download.
    st.image(
        st.session_state["plot_jpeg"],
        width="content",
    )

    # ========================================================
    # SLOPE COORDINATE OUTPUT
    # This information is displayed below the plot as text. It is not written
    # inside the plot itself.
    # ========================================================

    if "slope_output" in st.session_state:

        # These are the coordinates of the segment actually drawn after axis
        # clipping. Signed differences always use:
        #
        #     difference = ending point - starting point
        #
        # Therefore delta_x and delta_y can be positive or negative.
        slope_output = st.session_state["slope_output"]

        st.subheader("Slope coordinate output")

        st.write(
            "**X values:** "
            f"intersecting start = "
            f"{format_output_number(slope_output['x_start'])}; "
            f"intersecting end = "
            f"{format_output_number(slope_output['x_end'])}; "
            f"difference (end − start) = "
            f"{format_output_number(slope_output['delta_x'])}"
        )

        st.write(
            "**Y values:** "
            f"intersecting start = "
            f"{format_output_number(slope_output['y_start'])}; "
            f"intersecting end = "
            f"{format_output_number(slope_output['y_end'])}; "
            f"difference (end − start) = "
            f"{format_output_number(slope_output['delta_y'])}"
        )

        st.write(
            "**Fitted slope (ΔY/ΔX):** "
            f"{format_output_number(slope_output['slope'])}"
        )

    # The application intentionally exposes only one download format.
    st.download_button(
        "Download JPEG",
        data=st.session_state["plot_jpeg"],
        file_name="plot.jpg",
        mime="image/jpeg",
        width="stretch",
    )

else:

    # This is shown initially and whenever no successful plot exists in the
    # current Streamlit session.
    st.info("No plot has been created yet.")
