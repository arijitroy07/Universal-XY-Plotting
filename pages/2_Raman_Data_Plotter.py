"""Streamlit page for plotting and analysing Raman spectra.

This page adapts the supplied Jupyter Raman Viewer to Streamlit while keeping
its reusable calculations in raman_core.py. The plot is generated only after
the user selects Create / Update Raman Plot.
"""

from __future__ import annotations

import io
from pathlib import Path
import re
import sys

import matplotlib

# Streamlit Community Cloud runs without a desktop display. The noninteractive
# backend must be selected before importing pyplot or the Raman core module.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# Streamlit normally adds the repository root to Python's import path. This
# fallback also makes the page runnable directly during local testing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import raman_core as rc


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Raman Data Plotter",
    page_icon="🔬",
    layout="wide",
)

st.title("Raman Data Plotter")
st.caption(
    "Upload Raman spectra, display them as overlaid or vertically stacked "
    "traces, identify peaks, and export publication-ready figures."
)


# ============================================================
# SMALL INPUT AND EXPORT HELPERS
# ============================================================

def optional_number(label: str, key: str):
    """Return a floating-point value, or None when the field is blank."""
    raw_value = st.text_input(label, value="", key=key)

    if not raw_value.strip():
        return None

    try:
        return float(raw_value)
    except ValueError:
        st.warning(f"{label} must be numeric. Automatic scaling will be used.")
        return None


def parse_position_list(raw_value: str, field_name: str):
    """Parse comma-, semicolon-, or whitespace-separated Raman shifts."""
    raw_value = raw_value.strip()
    if not raw_value:
        return []

    values = []
    invalid = []

    for item in re.split(r"[,;\s]+", raw_value):
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            invalid.append(item)

    if invalid:
        st.warning(
            f"{field_name}: ignored nonnumeric value(s): {', '.join(invalid)}."
        )

    return values


def figure_bytes(fig, file_format: str, dpi: int):
    """Serialize a Matplotlib figure for a Streamlit download button."""
    buffer = io.BytesIO()
    save_options = {"format": file_format, "facecolor": "white"}

    if file_format in {"png", "tiff", "jpeg"}:
        save_options["dpi"] = dpi

    if file_format == "tiff":
        save_options["pil_kwargs"] = {"compression": "tiff_lzw"}

    fig.savefig(buffer, **save_options)
    buffer.seek(0)
    return buffer.getvalue()


def resolve_view(
    spectra,
    options,
    manual_x_min,
    manual_x_max,
    manual_y_min,
    manual_y_max,
):
    """Create plot limits and autoscale Y to data inside the visible X range."""
    full_view = rc.fit_view(spectra, options)

    display_x_start = (
        float(manual_x_min) if manual_x_min is not None else float(full_view[0])
    )
    display_x_end = (
        float(manual_x_max) if manual_x_max is not None else float(full_view[1])
    )

    if display_x_start == display_x_end:
        raise ValueError("X minimum and X maximum cannot be equal.")

    # Use ordered bounds only while selecting visible data. The original order
    # is restored in the final view so X minimum > X maximum reverses the axis.
    visible_x_low = min(display_x_start, display_x_end)
    visible_x_high = max(display_x_start, display_x_end)

    if rc.data_bounds(spectra, options, visible_x_low, visible_x_high) is None:
        raise ValueError("No visible Raman data exist inside the selected X range.")

    automatic_view = rc.fit_view(
        spectra,
        options,
        (visible_x_low, visible_x_high),
    )

    display_y_min = (
        float(manual_y_min) if manual_y_min is not None else float(automatic_view[2])
    )
    display_y_max = (
        float(manual_y_max) if manual_y_max is not None else float(automatic_view[3])
    )

    if display_y_min >= display_y_max:
        raise ValueError("Y minimum must be smaller than Y maximum.")

    return (
        display_x_start,
        display_x_end,
        display_y_min,
        display_y_max,
    )


# ============================================================
# SECTION 1 - LOAD RAMAN SPECTRA
# ============================================================

st.header("1. Upload Raman spectra")

uploaded_files = st.file_uploader(
    "Upload one or more Raman files",
    type=["dpt", "txt", "csv", "dat", "prn"],
    accept_multiple_files=True,
    help=(
        "Each file must contain at least two numeric columns: Raman shift and "
        "intensity. Header and comment lines are ignored automatically."
    ),
)

if not uploaded_files:
    st.info("Upload at least one Raman spectrum to continue.")
    st.stop()


# Remove a previously generated figure when the uploaded file collection
# changes. Normal setting changes retain the old figure until the plot button
# is selected again.
file_signature = tuple(
    (uploaded_file.name, len(uploaded_file.getvalue()))
    for uploaded_file in uploaded_files
)

if st.session_state.get("raman_file_signature") != file_signature:
    st.session_state["raman_file_signature"] = file_signature
    st.session_state.pop("raman_preview_png", None)
    st.session_state.pop("raman_figure_downloads", None)
    st.session_state.pop("raman_peak_table", None)
    st.session_state.pop("raman_peak_csv", None)
    st.session_state.pop("raman_download_base", None)


spectra = []
failed_files = []

for uploaded_file in uploaded_files:
    try:
        file_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
        spectrum = rc.spectrum_from_text(
            Path(uploaded_file.name).stem,
            file_text,
            spectra,
        )
        spectra.append(spectrum)
    except Exception as error:
        failed_files.append(f"{uploaded_file.name}: {error}")

for failure in failed_files:
    st.error(f"Could not read {failure}")

if not spectra:
    st.error("None of the uploaded files contained usable Raman data.")
    st.stop()

st.success(f"Loaded {len(spectra)} Raman spectrum/spectra.")


# ============================================================
# SECTION 2 - PER-SPECTRUM SETTINGS
# ============================================================

st.header("2. Configure spectra")
st.caption(
    "The first spectrum in the display order appears at the top of a stacked "
    "plot. Manual and excluded peak positions are entered in cm⁻¹."
)

configured_spectra = []

for spectrum_index, spectrum in enumerate(spectra):
    widget_key = f"raman_spectrum_{spectrum_index}_{spectrum.name}"

    with st.expander(
        f"{spectrum_index + 1}. {spectrum.name}",
        expanded=True,
    ):
        name_col, color_col, order_col, visibility_col = st.columns([3, 1, 1, 1])

        with name_col:
            spectrum.name = st.text_input(
                "Spectrum name",
                value=spectrum.name,
                key=f"{widget_key}_name",
            ).strip() or "Untitled"

        with color_col:
            spectrum.color = st.color_picker(
                "Trace color",
                value=spectrum.color,
                key=f"{widget_key}_color",
            )

        with order_col:
            display_order = int(
                st.number_input(
                    "Display order",
                    min_value=1,
                    max_value=max(1, len(spectra)),
                    value=spectrum_index + 1,
                    step=1,
                    key=f"{widget_key}_order",
                )
            )

        with visibility_col:
            spectrum.visible = st.checkbox(
                "Show",
                value=True,
                key=f"{widget_key}_visible",
            )

        peak_col1, peak_col2 = st.columns(2)

        with peak_col1:
            manual_positions = st.text_input(
                "Manual peak positions",
                value="",
                placeholder="Example: 383.5, 408.2",
                key=f"{widget_key}_manual_peaks",
                help=(
                    "Adds peaks at the nearest measured maxima to these Raman "
                    "shift positions. Separate multiple values with commas."
                ),
            )

        with peak_col2:
            excluded_positions = st.text_input(
                "Automatic peaks to exclude",
                value="",
                placeholder="Example: 520.1, 1002.4",
                key=f"{widget_key}_excluded_peaks",
                help=(
                    "Removes automatically detected peaks nearest to these "
                    "Raman shift positions."
                ),
            )

        spectrum.manual = parse_position_list(
            manual_positions,
            f"{spectrum.name} manual peaks",
        )

        # raman_core records excluded peaks using measured sample coordinates.
        # Convert user-entered positions to the nearest measured X values.
        excluded_values = parse_position_list(
            excluded_positions,
            f"{spectrum.name} excluded peaks",
        )
        spectrum.excluded = [
            float(spectrum.x[rc.nearest_index(spectrum.x, position)])
            for position in excluded_values
        ]

        st.caption(
            f"Full range: {spectrum.x_full[0]:g}–{spectrum.x_full[-1]:g} cm⁻¹ "
            f"· {len(spectrum.x_full):,} points"
        )

        configured_spectra.append(
            (display_order, spectrum_index, spectrum)
        )


# Sort first by the requested order and then by original upload order. The
# second value makes duplicate order numbers deterministic.
configured_spectra.sort(key=lambda item: (item[0], item[1]))
spectra = [item[2] for item in configured_spectra]


# ============================================================
# SECTION 3 - RAMAN RANGE, LAYOUT, TEXT, AND PEAK SETTINGS
# ============================================================

st.header("3. Configure Raman plot")

global_x_min = min(float(spectrum.x_full[0]) for spectrum in spectra)
global_x_max = max(float(spectrum.x_full[-1]) for spectrum in spectra)

st.subheader("Raman shift data range")

use_crop = st.checkbox(
    "Crop the spectra before peak detection",
    value=False,
    help=(
        "Cropping removes data outside the selected Raman shift interval from "
        "the plot, peak detection, peak table, and exported figure."
    ),
)

crop_low = global_x_min
crop_high = global_x_max

if use_crop:
    crop_col1, crop_col2 = st.columns(2)

    with crop_col1:
        crop_low = st.number_input(
            "Crop start (cm⁻¹)",
            value=float(global_x_min),
            format="%.6f",
        )

    with crop_col2:
        crop_high = st.number_input(
            "Crop end (cm⁻¹)",
            value=float(global_x_max),
            format="%.6f",
        )


st.subheader("Layout and spacing")

layout_col1, layout_col2 = st.columns(2)

with layout_col1:
    layout_choice = st.radio(
        "Spectrum layout",
        ["Stacked", "Overlay"],
        horizontal=True,
    )

with layout_col2:
    stack_spacing = st.number_input(
        "Stack spacing (% of tallest intensity range)",
        min_value=0.0,
        max_value=1000.0,
        value=45.0,
        step=5.0,
        disabled=layout_choice != "Stacked",
    )


st.subheader("Titles and labels")

graph_title = st.text_input(
    "Graph title",
    value="",
    placeholder="Leave blank for no title",
)

text_col1, text_col2, text_col3, text_col4 = st.columns(4)

with text_col1:
    show_graph_title = st.checkbox(
        "Show graph title",
        value=True,
    )

with text_col2:
    show_axis_titles = st.checkbox(
        "Show axis titles",
        value=True,
    )

with text_col3:
    show_spectrum_names = st.checkbox(
        "Show spectrum names",
        value=True,
    )

with text_col4:
    show_peak_labels = st.checkbox(
        "Label detected peaks",
        value=True,
    )


st.subheader("Peak detection")

peak_col1, peak_col2, peak_col3 = st.columns(3)

with peak_col1:
    detect_peaks_automatically = st.checkbox(
        "Detect peaks automatically",
        value=True,
    )

with peak_col2:
    minimum_prominence = st.number_input(
        "Minimum prominence (% of spectrum range)",
        min_value=0.01,
        max_value=100.0,
        value=3.0,
        step=0.25,
        disabled=not detect_peaks_automatically,
    )

with peak_col3:
    minimum_spacing = st.number_input(
        "Minimum peak spacing (cm⁻¹)",
        min_value=0.01,
        max_value=1000.0,
        value=8.0,
        step=1.0,
        disabled=not detect_peaks_automatically,
    )


st.subheader("Visible axis range")
st.caption(
    "Leave fields blank for automatic limits. Y automatically fits the data "
    "visible inside a customized X range. Enter X minimum greater than X "
    "maximum to reverse the Raman-shift axis."
)

axis_col1, axis_col2, axis_col3, axis_col4 = st.columns(4)

with axis_col1:
    x_minimum = optional_number(
        "X minimum",
        "raman_x_minimum",
    )

with axis_col2:
    x_maximum = optional_number(
        "X maximum",
        "raman_x_maximum",
    )

with axis_col3:
    y_minimum = optional_number(
        "Y minimum",
        "raman_y_minimum",
    )

with axis_col4:
    y_maximum = optional_number(
        "Y maximum",
        "raman_y_maximum",
    )


st.subheader("Publication figure settings")

figure_col1, figure_col2, figure_col3, figure_col4 = st.columns(4)

with figure_col1:
    figure_width_name = st.selectbox(
        "Figure width",
        list(rc.FIGURE_WIDTHS.keys()),
        index=0,
    )

with figure_col2:
    figure_aspect_name = st.selectbox(
        "Aspect ratio",
        list(rc.ASPECTS.keys()),
        index=0,
    )

with figure_col3:
    raster_dpi = st.selectbox(
        "Raster resolution",
        [300, 600, 1200],
        index=1,
        format_func=lambda value: f"{value} dpi",
    )

with figure_col4:
    output_name = st.text_input(
        "Download file name",
        value="raman-figure",
    )


# ============================================================
# SECTION 4 - EXPLICIT PLOT GENERATION
# ============================================================

st.header("4. Create Raman plot")
st.caption(
    "Changing a setting does not redraw the figure. Select the button below "
    "when you are ready to create or update it."
)

create_plot = st.button(
    "Create / Update Raman Plot",
    type="primary",
    width="stretch",
)

if create_plot:
    st.session_state.pop("raman_preview_png", None)
    st.session_state.pop("raman_figure_downloads", None)
    st.session_state.pop("raman_peak_table", None)
    st.session_state.pop("raman_peak_csv", None)
    st.session_state.pop("raman_download_base", None)

    generated_figure = None

    try:
        visible_spectra = [spectrum for spectrum in spectra if spectrum.visible]

        if not visible_spectra:
            raise ValueError("Select at least one spectrum to display.")

        if use_crop:
            crop_start = min(float(crop_low), float(crop_high))
            crop_end = max(float(crop_low), float(crop_high))

            if crop_start == crop_end:
                raise ValueError("Crop start and crop end cannot be equal.")

            if not any(
                rc.points_in_range(spectrum, crop_start, crop_end) >= 3
                for spectrum in spectra
            ):
                raise ValueError(
                    "No spectrum contains at least three points inside the "
                    "selected crop range."
                )

            rc.apply_crop(
                spectra,
                (crop_start, crop_end),
            )
        else:
            rc.apply_crop(spectra, None)

        peak_settings = rc.PeakSettings(
            auto=detect_peaks_automatically,
            min_prominence_pct=float(minimum_prominence),
            min_spacing=float(minimum_spacing),
        )

        plot_options = rc.PlotOptions(
            layout=layout_choice.lower(),
            offset_pct=float(stack_spacing),
            labels=show_peak_labels,
            title=graph_title.strip(),
            show_title=show_graph_title,
            show_axis_titles=show_axis_titles,
            show_names=show_spectrum_names,
        )

        rc.compute_all_peaks(
            spectra,
            peak_settings,
        )

        plot_view = resolve_view(
            spectra,
            plot_options,
            x_minimum,
            x_maximum,
            y_minimum,
            y_maximum,
        )

        figure_width_cm, figure_scale = rc.FIGURE_WIDTHS[figure_width_name]
        figure_aspect = rc.ASPECTS[figure_aspect_name]

        generated_figure = rc.journal_figure(
            spectra,
            plot_options,
            plot_view,
            width_cm=figure_width_cm,
            aspect=figure_aspect,
            scale=figure_scale,
        )

        # A moderate-resolution PNG is stored for the on-screen preview. The
        # download files below use the selected raster resolution.
        preview_png = figure_bytes(
            generated_figure,
            "png",
            dpi=180,
        )

        figure_downloads = {
            "png": figure_bytes(generated_figure, "png", raster_dpi),
            "tif": figure_bytes(generated_figure, "tiff", raster_dpi),
            "svg": figure_bytes(generated_figure, "svg", raster_dpi),
            "pdf": figure_bytes(generated_figure, "pdf", raster_dpi),
        }

        peak_dataframe = rc.peak_table(spectra)
        peak_csv = peak_dataframe.to_csv(index=False).encode("utf-8")

        download_base = rc.safe_filename(
            output_name.strip() or graph_title.strip(),
            fallback="raman-figure",
        )

        st.session_state["raman_preview_png"] = preview_png
        st.session_state["raman_figure_downloads"] = figure_downloads
        st.session_state["raman_peak_table"] = peak_dataframe
        st.session_state["raman_peak_csv"] = peak_csv
        st.session_state["raman_download_base"] = download_base

    except Exception as error:
        st.error(f"The Raman plot could not be created: {error}")

    finally:
        if generated_figure is not None:
            plt.close(generated_figure)


# ============================================================
# PERSISTED FIGURE, PEAK TABLE, AND DOWNLOADS
# ============================================================

if "raman_preview_png" in st.session_state:
    st.image(
        st.session_state["raman_preview_png"],
        width="content",
        caption="Last generated Raman plot",
    )

    st.subheader("Detected peak table")

    peak_table = st.session_state["raman_peak_table"]

    if peak_table.empty:
        st.info(
            "No peaks were detected. Lower the minimum prominence or enter "
            "manual peak positions, then update the plot."
        )
    else:
        st.dataframe(
            peak_table,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Downloads")

    download_base = st.session_state["raman_download_base"]
    downloads = st.session_state["raman_figure_downloads"]
    download_col1, download_col2, download_col3, download_col4 = st.columns(4)

    with download_col1:
        st.download_button(
            "Download PNG",
            data=downloads["png"],
            file_name=f"{download_base}.png",
            mime="image/png",
            width="stretch",
        )

    with download_col2:
        st.download_button(
            "Download TIFF",
            data=downloads["tif"],
            file_name=f"{download_base}.tif",
            mime="image/tiff",
            width="stretch",
        )

    with download_col3:
        st.download_button(
            "Download SVG",
            data=downloads["svg"],
            file_name=f"{download_base}.svg",
            mime="image/svg+xml",
            width="stretch",
        )

    with download_col4:
        st.download_button(
            "Download PDF",
            data=downloads["pdf"],
            file_name=f"{download_base}.pdf",
            mime="application/pdf",
            width="stretch",
        )

    if not peak_table.empty:
        st.download_button(
            "Download Peak Table CSV",
            data=st.session_state["raman_peak_csv"],
            file_name=f"{download_base}-peaks.csv",
            mime="text/csv",
            width="stretch",
        )

else:
    st.info("No Raman plot has been created yet.")
