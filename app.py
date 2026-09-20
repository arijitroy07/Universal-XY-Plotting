from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from data_reader import (
    clean_numeric_series,
    get_excel_sheets,
    prepare_table,
    read_uploaded_file,
)
from plotting import (
    add_dataset_to_axis,
    apply_axis_settings,
    figure_bytes,
    parse_optional_float,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Universal XY Plotter",
    page_icon="📈",
    layout="wide",
)

st.title("Universal XY Data Plotter")

st.caption(
    "Upload XLS, XLSX, CSV, TXT, or XY files. Select X and Y columns "
    "for each dataset, then customize and export the plot."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def optional_number(label, key):
    """
    Display a text input that returns either a float or None.
    """
    raw_value = st.text_input(
        label,
        value="",
        key=key,
    )

    value = parse_optional_float(raw_value)

    if raw_value.strip() and value is None:
        st.warning(
            f"{label} must be a number. "
            "Automatic scaling will be used."
        )

    return value


def visible_x_mask(
    x_values,
    x_min=None,
    x_max=None,
):
    """
    Select points inside the visible X-axis range.

    X minimum may be greater than X maximum when the
    X-axis is reversed.
    """
    x_values = np.asarray(
        x_values,
        dtype=float,
    )

    mask = np.isfinite(x_values)

    if (
        x_min is not None
        and x_max is not None
    ):
        lower_x = min(
            x_min,
            x_max,
        )

        upper_x = max(
            x_min,
            x_max,
        )

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
    Calculate Y limits using only data visible inside
    the selected X-axis range.

    xy_datasets must contain:
        [(x_array, y_array), ...]
    """
    visible_values = []

    for x_values, y_values in xy_datasets:
        x_values = np.asarray(
            x_values,
            dtype=float,
        )

        y_values = np.asarray(
            y_values,
            dtype=float,
        )

        mask = visible_x_mask(
            x_values,
            x_min,
            x_max,
        )

        mask &= np.isfinite(y_values)

        if np.any(mask):
            visible_values.append(
                y_values[mask]
            )

    if not visible_values:
        return None, None

    combined_y = np.concatenate(
        visible_values
    )

    minimum_y = float(
        np.min(combined_y)
    )

    maximum_y = float(
        np.max(combined_y)
    )

    y_range = maximum_y - minimum_y

    if y_range == 0:
        if minimum_y == 0:
            padding = 1.0
        else:
            padding = max(
                abs(minimum_y) * padding_fraction,
                1e-9,
            )
    else:
        padding = (
            y_range
            * padding_fraction
        )

    return (
        minimum_y - padding,
        maximum_y + padding,
    )


def resolve_y_limits(
    xy_datasets,
    x_min,
    x_max,
    manual_y_min,
    manual_y_max,
):
    """
    Use manual Y limits when provided.

    If the X range is customized and either Y limit is
    blank, calculate the missing limit from the visible data.
    """
    effective_y_min = manual_y_min
    effective_y_max = manual_y_max

    if (
        x_min is None
        and x_max is None
    ):
        return (
            effective_y_min,
            effective_y_max,
        )

    auto_y_min, auto_y_max = visible_y_limits(
        xy_datasets,
        x_min,
        x_max,
    )

    if effective_y_min is None:
        effective_y_min = auto_y_min

    if effective_y_max is None:
        effective_y_max = auto_y_max

    return (
        effective_y_min,
        effective_y_max,
    )


def normalize_visible_trace(
    y_values,
    x_values,
    x_min=None,
    x_max=None,
):
    """
    Normalize one trace from 0 to 1.

    When the X range is customized, the normalization minimum
    and maximum are obtained from only the visible section.
    """
    x_values = np.asarray(
        x_values,
        dtype=float,
    )

    y_values = np.asarray(
        y_values,
        dtype=float,
    )

    reference_values = y_values

    if (
        x_min is not None
        or x_max is not None
    ):
        mask = visible_x_mask(
            x_values,
            x_min,
            x_max,
        )

        mask &= np.isfinite(y_values)

        if np.any(mask):
            reference_values = y_values[mask]

    finite_reference = reference_values[
        np.isfinite(reference_values)
    ]

    if finite_reference.size == 0:
        return np.zeros_like(
            y_values
        )

    minimum_y = np.min(
        finite_reference
    )

    maximum_y = np.max(
        finite_reference
    )

    if maximum_y == minimum_y:
        return np.full_like(
            y_values,
            0.5,
        )

    return (
        (y_values - minimum_y)
        / (maximum_y - minimum_y)
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("1. Upload data")

uploaded_files = st.file_uploader(
    "Upload one or more data files",
    type=[
        "xls",
        "xlsx",
        "csv",
        "txt",
        "xy",
    ],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info(
        "Upload at least one data file to continue."
    )

    st.stop()


# ============================================================
# DATASET AND COLUMN SELECTION
# ============================================================

st.header("2. Select X and Y data")

datasets = []


for file_index, uploaded_file in enumerate(
    uploaded_files
):
    file_key = (
        f"file_{file_index}_"
        f"{uploaded_file.name}"
    )

    with st.expander(
        f"{file_index + 1}. {uploaded_file.name}",
        expanded=True,
    ):
        extension = Path(
            uploaded_file.name
        ).suffix.lower()

        selected_sheet = None

        try:
            if extension in {
                ".xls",
                ".xlsx",
            }:
                sheets = get_excel_sheets(
                    uploaded_file.getvalue()
                )

                selected_sheet = st.selectbox(
                    "Excel sheet",
                    sheets,
                    key=f"sheet_{file_key}",
                )

            raw_df, detected_separator = (
                read_uploaded_file(
                    uploaded_file,
                    selected_sheet,
                )
            )

        except Exception as error:
            st.error(
                f"Could not read this file: {error}"
            )

            continue

        with st.expander(
            "Detection settings and raw preview"
        ):
            st.dataframe(
                raw_df.head(20),
                width="stretch",
            )

            detection_mode = st.radio(
                "Table detection",
                [
                    "Automatic",
                    "Manual override",
                ],
                horizontal=True,
                key=f"detection_{file_key}",
            )

            data_start_override = None
            header_row_override = None
            unit_row_override = None

            if detection_mode == "Manual override":
                st.caption(
                    "Rows are numbered from 1. "
                    "Use 0 for no header or units row."
                )

                row_col1, row_col2, row_col3 = (
                    st.columns(3)
                )

                with row_col1:
                    data_start_override = int(
                        st.number_input(
                            "First numeric data row",
                            min_value=1,
                            max_value=max(
                                1,
                                len(raw_df),
                            ),
                            value=min(
                                2,
                                max(
                                    1,
                                    len(raw_df),
                                ),
                            ),
                            step=1,
                            key=(
                                f"data_start_"
                                f"{file_key}"
                            ),
                        )
                    ) - 1

                with row_col2:
                    header_choice = int(
                        st.number_input(
                            "Header row (0 = none)",
                            min_value=0,
                            max_value=max(
                                0,
                                len(raw_df),
                            ),
                            value=(
                                1
                                if len(raw_df)
                                else 0
                            ),
                            step=1,
                            key=(
                                f"header_row_"
                                f"{file_key}"
                            ),
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
                            max_value=max(
                                0,
                                len(raw_df),
                            ),
                            value=0,
                            step=1,
                            key=(
                                f"unit_row_"
                                f"{file_key}"
                            ),
                        )
                    )

                    unit_row_override = (
                        unit_choice - 1
                        if unit_choice
                        else None
                    )

        try:
            prepared_df, metadata = (
                prepare_table(
                    raw_df,
                    data_start_override=(
                        data_start_override
                    ),
                    header_row_override=(
                        header_row_override
                    ),
                    unit_row_override=(
                        unit_row_override
                    ),
                )
            )

        except Exception as error:
            st.error(
                "Could not detect a usable table: "
                f"{error}"
            )

            continue

        if prepared_df.shape[1] < 2:
            st.warning(
                "At least two usable numeric columns "
                "could not be detected."
            )

            continue

        detected_columns = list(
            prepared_df.columns
        )

        detection_text = (
            "Numeric data begins near row "
            f"{metadata['data_start'] + 1}."
        )

        if metadata["header_row"] is not None:
            detection_text += (
                " Header row: "
                f"{metadata['header_row'] + 1}."
            )

        if metadata["unit_row"] is not None:
            detection_text += (
                " Units row: "
                f"{metadata['unit_row'] + 1}."
            )

        if detected_separator:
            detection_text += (
                " Detected separator: "
                f"{detected_separator!r}."
            )

        st.caption(
            detection_text
        )

        st.write(
            "**Cleaned preview**"
        )

        st.dataframe(
            prepared_df.head(10),
            width="stretch",
        )

        selection_col1, selection_col2 = (
            st.columns(2)
        )

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
                index=min(
                    1,
                    len(detected_columns) - 1,
                ),
                key=f"y_{file_key}",
            )

        label_col, color_col = st.columns(
            [2, 1]
        )

        with label_col:
            legend_label = st.text_input(
                "Legend label",
                value=Path(
                    uploaded_file.name
                ).stem,
                key=f"legend_{file_key}",
            )

        with color_col:
            use_custom_color = st.checkbox(
                "Custom color",
                value=False,
                key=f"use_color_{file_key}",
            )

            if use_custom_color:
                color = st.color_picker(
                    "Trace color",
                    "#1f77b4",
                    key=f"color_{file_key}",
                )
            else:
                color = None

        x_numeric = clean_numeric_series(
            prepared_df[x_column]
        )

        y_numeric = clean_numeric_series(
            prepared_df[y_column]
        )

        valid = (
            x_numeric.notna()
            & y_numeric.notna()
        )

        x_numeric = x_numeric[valid]
        y_numeric = y_numeric[valid]

        if x_numeric.empty:
            st.warning(
                "No valid numeric X-Y pairs were "
                "found for these columns."
            )

            continue

        st.success(
            f"{len(x_numeric):,} valid "
            "X-Y points detected."
        )

        datasets.append(
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
    st.error(
        "No usable datasets are available."
    )

    st.stop()


# ============================================================
# PLOT CONFIGURATION
# ============================================================

st.header("3. Configure plot")

layout_mode = st.radio(
    "Multiple-file plotting mode",
    [
        "Overlap",
        "Stacked traces with Y offset",
        "Stacked panels",
    ],
    horizontal=True,
)

plot_style = st.radio(
    "Plot type",
    [
        "Line",
        "Scatter",
        "Scatter + Line",
    ],
    horizontal=True,
)


# ============================================================
# FIGURE AND TRACE STYLE
# ============================================================

st.subheader(
    "Figure and trace style"
)

size_col1, size_col2, style_col1, style_col2 = (
    st.columns(4)
)

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
    "Current aspect ratio: "
    f"{figure_width / figure_height:.2f}:1"
)


# ============================================================
# TITLES AND LABELS
# ============================================================

st.subheader(
    "Titles and labels"
)

plot_title = st.text_input(
    "Plot title",
    value="",
)

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

legend_title = st.text_input(
    "Legend title",
    value="",
)


# ============================================================
# AXIS SETTINGS
# ============================================================

st.subheader(
    "Axis limits and tick spacing"
)

st.caption(
    "Leave a field blank to use automatic values."
)

if layout_mode == "Stacked traces with Y offset":
    axis_col1, axis_col2, axis_col3 = (
        st.columns(3)
    )

    with axis_col1:
        x_min = optional_number(
            "X minimum",
            "x_min",
        )

    with axis_col2:
        x_max = optional_number(
            "X maximum",
            "x_max",
        )

    with axis_col3:
        x_tick = optional_number(
            "X major tick interval",
            "x_tick",
        )

    y_min = None
    y_max = None
    y_tick = None

    st.info(
        "Every offset-stacked trace is automatically "
        "scaled using its own Y minimum and maximum "
        "within the visible X range. Y-axis numbers "
        "are hidden."
    )

else:
    axis_cols = st.columns(4)

    with axis_cols[0]:
        x_min = optional_number(
            "X minimum",
            "x_min",
        )

    with axis_cols[1]:
        x_max = optional_number(
            "X maximum",
            "x_max",
        )

    with axis_cols[2]:
        y_min = optional_number(
            "Y minimum",
            "y_min",
        )

    with axis_cols[3]:
        y_max = optional_number(
            "Y maximum",
            "y_max",
        )

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
    "To reverse the X-axis, enter an X minimum "
    "greater than the X maximum. When Y limits are "
    "blank, they automatically fit the data visible "
    "inside the selected X range."
)


# X minimum may be larger than X maximum.
# Only identical values are invalid.
if (
    x_min is not None
    and x_max is not None
    and x_min == x_max
):
    st.error(
        "X minimum and X maximum cannot be equal."
    )

    st.stop()


if (
    y_min is not None
    and y_max is not None
    and y_min >= y_max
):
    st.error(
        "Y minimum must be smaller than Y maximum."
    )

    st.stop()


# ============================================================
# LEGEND AND GRID
# ============================================================

option_col1, option_col2, option_col3 = (
    st.columns(3)
)

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
# STACKING OPTIONS
# ============================================================

fixed_panel_y = False
trace_gap = 0.25

if layout_mode == "Stacked traces with Y offset":
    trace_gap = st.slider(
        "Space between stacked traces",
        min_value=0.0,
        max_value=2.0,
        value=0.25,
        step=0.05,
    )

elif layout_mode == "Stacked panels":
    fixed_panel_y = st.checkbox(
        "Use one shared Y-axis range for all panels",
        value=False,
    )


# ============================================================
# CREATE OR UPDATE PLOT
# ============================================================

st.header("4. Plot")

st.caption(
    "Changing a setting does not redraw the figure. "
    "Select the button below when you are ready to "
    "create or update it."
)

create_plot = st.button(
    "Create / Update Plot",
    type="primary",
    width="stretch",
)


if create_plot:
    # Remove the previous image while generating the update.
    st.session_state.pop(
        "plot_jpeg",
        None,
    )

    fig = None

    try:
        # ----------------------------------------------------
        # OVERLAPPING DATASETS
        # ----------------------------------------------------

        if layout_mode == "Overlap":
            fig, ax = plt.subplots(
                figsize=(
                    figure_width,
                    figure_height,
                )
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

            effective_y_min, effective_y_max = (
                resolve_y_limits(
                    [
                        (
                            dataset["x"],
                            dataset["y"],
                        )
                        for dataset in datasets
                    ],
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                )
            )

            ax.set_xlabel(
                x_axis_label
            )

            ax.set_ylabel(
                y_axis_label
            )

            if plot_title:
                ax.set_title(
                    plot_title
                )

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
                    title=(
                        legend_title
                        if legend_title
                        else None
                    ),
                    loc=legend_location,
                )

            fig.tight_layout()


        # ----------------------------------------------------
        # OFFSET-STACKED TRACES
        # ----------------------------------------------------

        elif layout_mode == "Stacked traces with Y offset":
            fig, ax = plt.subplots(
                figsize=(
                    figure_width,
                    figure_height,
                )
            )

            trace_step = (
                1.0
                + trace_gap
            )

            for dataset_index, dataset in enumerate(
                datasets
            ):
                independently_scaled_y = (
                    normalize_visible_trace(
                        y_values=dataset["y"],
                        x_values=dataset["x"],
                        x_min=x_min,
                        x_max=x_max,
                    )
                )

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

            ax.set_xlabel(
                x_axis_label
            )

            ax.set_ylabel(
                y_axis_label
            )

            if plot_title:
                ax.set_title(
                    plot_title
                )

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

            ax.set_ylim(
                -0.05,
                (
                    (len(datasets) - 1)
                    * trace_step
                    + 1.05
                ),
            )

            # Hide Y-axis numbers and ticks.
            # The Y-axis label remains visible.
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                labelleft=False,
            )

            ax.set_yticks([])

            if show_legend:
                ax.legend(
                    title=(
                        legend_title
                        if legend_title
                        else None
                    ),
                    loc=legend_location,
                )

            fig.tight_layout()


        # ----------------------------------------------------
        # STACKED PANELS
        # ----------------------------------------------------

        else:
            number_of_plots = len(
                datasets
            )

            fig, axes = plt.subplots(
                number_of_plots,
                1,
                figsize=(
                    figure_width,
                    figure_height,
                ),
                sharex=True,
                squeeze=False,
            )

            axes = axes.ravel()

            common_y_min = None
            common_y_max = None

            if fixed_panel_y:
                (
                    automatic_common_y_min,
                    automatic_common_y_max,
                ) = visible_y_limits(
                    [
                        (
                            dataset["x"],
                            dataset["y"],
                        )
                        for dataset in datasets
                    ],
                    x_min,
                    x_max,
                )

                common_y_min = (
                    y_min
                    if y_min is not None
                    else automatic_common_y_min
                )

                common_y_max = (
                    y_max
                    if y_max is not None
                    else automatic_common_y_max
                )

            for ax, dataset in zip(
                axes,
                datasets,
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
                    panel_y_min = (
                        common_y_min
                    )

                    panel_y_max = (
                        common_y_max
                    )

                else:
                    (
                        panel_y_min,
                        panel_y_max,
                    ) = resolve_y_limits(
                        [
                            (
                                dataset["x"],
                                dataset["y"],
                            )
                        ],
                        x_min,
                        x_max,
                        y_min,
                        y_max,
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
                        title=(
                            legend_title
                            if legend_title
                            else None
                        ),
                        loc=legend_location,
                    )

            axes[-1].set_xlabel(
                x_axis_label
            )

            fig.supylabel(
                y_axis_label
            )

            if plot_title:
                fig.suptitle(
                    plot_title
                )

                fig.tight_layout(
                    rect=[
                        0,
                        0,
                        1,
                        0.96,
                    ]
                )

            else:
                fig.tight_layout()


        # ----------------------------------------------------
        # SAVE GENERATED PLOT
        # ----------------------------------------------------

        st.session_state["plot_jpeg"] = (
            figure_bytes(
                fig,
                "jpeg",
                dpi=300,
            )
        )

    except Exception as error:
        st.error(
            "The plot could not be created: "
            f"{error}"
        )

    finally:
        if fig is not None:
            plt.close(fig)


# ============================================================
# DISPLAY AND DOWNLOAD
# ============================================================

if "plot_jpeg" in st.session_state:
    st.image(
        st.session_state["plot_jpeg"],
        width="content",
    )

    st.download_button(
        "Download JPEG",
        data=st.session_state["plot_jpeg"],
        file_name="plot.jpg",
        mime="image/jpeg",
        width="stretch",
    )

else:
    st.info(
        "No plot has been created yet."
    )
