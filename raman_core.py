r"""Core of Raman Viewer: reading OPUS .dpt files, finding peaks and drawing figures.

Everything here works without Jupyter. The notebook UI (raman_ui.py) is a thin layer
on top of these functions, so anything you can do with the mouse you can also script:

    import raman_core as rc
    rc.use_style()
    spectra = rc.load_folder(r"C:\Data\Raman")
    rc.compute_all_peaks(spectra, rc.PeakSettings(min_prominence_pct=2))
    view = rc.fit_view(spectra, rc.PlotOptions())
    rc.export_figure("figure.png", spectra, rc.PlotOptions(), view, width_cm=8.5, dpi=600)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullFormatter, ScalarFormatter

PALETTE = ["#3566D6", "#D6455A", "#12948A", "#D98B0B", "#8052C8", "#4A8F2E", "#C24F8C", "#56697F"]
FONT = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

# Standard journal figure widths (cm) and a type/stroke scale for each.
FIGURE_WIDTHS = {
    "Single column · 8.5 cm": (8.5, 1.0),
    "1.5 column · 11.4 cm": (11.4, 1.0),
    "Double column · 17.8 cm": (17.8, 1.05),
    "Slide · 25.4 cm": (25.4, 1.75),
}
ASPECTS = {"4 : 3": 0.75, "3 : 2": 2 / 3, "16 : 9": 0.5625, "Square": 1.0}


def use_style() -> None:
    """Fonts and file settings for publication output (Arial, editable text in SVG/PDF)."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "mathtext.default": "regular",  # the cm^-1 superscript is set in Arial, not Computer Modern
        "axes.unicode_minus": True,
        "svg.fonttype": "none",         # SVG text stays text, editable in Illustrator/Inkscape
        "pdf.fonttype": 42,             # embed TrueType, as most journals require
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------- data

@dataclass(eq=False)
class Peak:
    i: int                 # index of the highest sample
    x: float               # refined centre, cm-1
    y: float               # refined height
    prominence: float
    fwhm: float | None     # full width at half prominence, cm-1
    source: str            # "auto" or "manual"


@dataclass(eq=False)
class Spectrum:
    name: str
    x: np.ndarray
    y: np.ndarray
    color: str = PALETTE[0]
    visible: bool = True
    path: Path | None = None
    manual: list[float] = field(default_factory=list)    # x of hand-picked peaks
    excluded: list[float] = field(default_factory=list)  # x of auto peaks the user removed
    peaks: list[Peak] = field(default_factory=list)
    # The complete data as read from the file; x and y may be a cropped part of it (apply_crop).
    x_full: np.ndarray | None = None
    y_full: np.ndarray | None = None

    def __post_init__(self):
        if self.x_full is None:
            self.x_full, self.y_full = self.x, self.y

    @property
    def in_range(self) -> bool:
        """False when the current crop leaves fewer than 3 points of this spectrum."""
        return len(self.x) >= 3

    @property
    def shown(self) -> bool:
        return self.visible and self.in_range

    @property
    def ymin(self) -> float:
        return float(self.y.min()) if len(self.y) else 0.0

    @property
    def ymax(self) -> float:
        return float(self.y.max()) if len(self.y) else 0.0

    @property
    def yrange(self) -> float:
        return (self.ymax - self.ymin) or 1.0


@dataclass
class PeakSettings:
    auto: bool = True
    min_prominence_pct: float = 3.0  # share of each spectrum's intensity range
    min_spacing: float = 8.0         # cm-1


@dataclass
class PlotOptions:
    layout: str = "stacked"          # "stacked" or "overlay"
    offset_pct: float = 45.0         # stack spacing, % of the tallest spectrum's range
    labels: bool = True              # wavenumber labels on peaks
    title: str = ""
    show_title: bool = True
    show_axis_titles: bool = True
    show_names: bool = True          # stack labels or legend


# --------------------------------------------------------------------------- reading

def parse_dpt(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Two numeric columns (shift, intensity) separated by tabs, semicolons, commas or spaces.
    Header or comment lines are skipped; rows are sorted by shift."""
    xs, ys = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.search(r"[\t;]", line):
            parts = [p.strip().replace(",", ".") for p in re.split(r"[\t;]+", line)]
        else:
            parts = re.split(r"[,\s]+", line)
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
    if len(xs) < 3:
        raise ValueError("fewer than 3 numeric rows; expected two columns: shift and intensity")
    order = np.argsort(xs, kind="stable")
    return np.asarray(xs)[order], np.asarray(ys)[order]


def _next_color(existing: list[Spectrum]) -> str:
    used = {s.color.lower() for s in existing}
    for c in PALETTE:
        if c.lower() not in used:
            return c
    return PALETTE[len(existing) % len(PALETTE)]


def read_dpt(path, existing: list[Spectrum] | None = None) -> Spectrum:
    path = Path(path)
    x, y = parse_dpt(path.read_text(encoding="utf-8", errors="replace"))
    return Spectrum(name=path.stem, x=x, y=y, color=_next_color(existing or []), path=path)


def spectrum_from_text(name: str, text: str, existing: list[Spectrum] | None = None) -> Spectrum:
    x, y = parse_dpt(text)
    return Spectrum(name=name, x=x, y=y, color=_next_color(existing or []))


def apply_crop(spectra: list[Spectrum], crop: tuple[float, float] | None) -> None:
    """Limit every spectrum to the Raman shift range crop=(lo, hi) cm-1, or restore the full
    data with None. Recompute peaks afterwards."""
    for s in spectra:
        if crop is None:
            s.x, s.y = s.x_full, s.y_full
        else:
            m = (s.x_full >= crop[0]) & (s.x_full <= crop[1])
            s.x, s.y = s.x_full[m], s.y_full[m]


def points_in_range(s: Spectrum, lo: float, hi: float) -> int:
    return int(np.count_nonzero((s.x_full >= lo) & (s.x_full <= hi)))


def load_folder(folder, pattern: str = "*.dpt", recursive: bool = True) -> list[Spectrum]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}. Check the path to your .dpt files.")
    files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    spectra: list[Spectrum] = []
    for f in files:
        spectra.append(read_dpt(f, spectra))
    return spectra


# --------------------------------------------------------------------------- peaks

def prominence(y: np.ndarray, i: int) -> float:
    """Height above the higher of the two lowest points reached before meeting taller data."""
    h = y[i]

    def side_min(side):
        stop = np.nonzero(side > h)[0]
        seg = side[: stop[0]] if stop.size else side  # walk until the data rises above the peak
        return min(h, seg.min()) if seg.size else h   # empty when the very next point is taller

    return float(h - max(side_min(y[:i][::-1]), side_min(y[i + 1:])))


def detect_peaks(sp: Spectrum, settings: PeakSettings) -> list[int]:
    y, x = sp.y, sp.x
    min_prom = settings.min_prominence_pct / 100 * sp.yrange
    idx = np.nonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    cands = [(i, p) for i in idx if (p := prominence(y, int(i))) >= min_prom]
    cands.sort(key=lambda c: -c[1])  # most prominent wins the spacing contest
    kept: list[int] = []
    for i, _ in cands:
        if all(abs(x[k] - x[i]) >= settings.min_spacing for k in kept):
            kept.append(int(i))
    return kept


def refine(sp: Spectrum, i: int) -> tuple[float, float]:
    """Sub-sample peak centre by a parabola through the three top points."""
    x, y = sp.x, sp.y
    if i <= 0 or i >= len(y) - 1:
        return float(x[i]), float(y[i])
    a, b, c = y[i - 1], y[i], y[i + 1]
    d = a - 2 * b + c
    if d >= 0 or b < a or b < c:  # only a true local maximum has a vertex within half a sample
        return float(x[i]), float(b)
    p = 0.5 * (a - c) / d
    return float(x[i] + p * (x[i + 1] - x[i - 1]) / 2), float(b - 0.25 * (a - c) * p)


def fwhm(sp: Spectrum, i: int, prom: float) -> float | None:
    x, y = sp.x, sp.y
    half = y[i] - prom / 2
    j = i
    while j > 0 and y[j] > half:
        j -= 1
    if y[j] > half:
        return None
    xl = x[j] + (half - y[j]) * (x[j + 1] - x[j]) / (y[j + 1] - y[j])
    k = i
    while k < len(y) - 1 and y[k] > half:
        k += 1
    if y[k] > half:
        return None
    xr = x[k - 1] + (half - y[k - 1]) * (x[k] - x[k - 1]) / (y[k] - y[k - 1])
    return float(xr - xl)


def make_peak(sp: Spectrum, i: int, source: str) -> Peak:
    cx, cy = refine(sp, i)
    prom = prominence(sp.y, i)
    return Peak(i=i, x=cx, y=cy, prominence=prom, fwhm=fwhm(sp, i, prom), source=source)


def nearest_index(x: np.ndarray, v: float) -> int:
    return int(np.clip(np.abs(x - v).argmin(), 0, len(x) - 1))


def _near(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


def compute_peaks(sp: Spectrum, settings: PeakSettings) -> None:
    if not sp.in_range:
        sp.peaks = []
        return
    out: list[Peak] = []
    if settings.auto:
        for i in detect_peaks(sp, settings):
            if not any(_near(e, sp.x[i]) for e in sp.excluded):
                out.append(make_peak(sp, i, "auto"))
    for mx in sp.manual:
        if not (sp.x[0] <= mx <= sp.x[-1]):
            continue  # outside the current crop; kept for when the range is widened again
        i = nearest_index(sp.x, mx)
        if not any(p.i == i for p in out):
            out.append(make_peak(sp, i, "manual"))
    sp.peaks = sorted(out, key=lambda p: p.x)


def compute_all_peaks(spectra: list[Spectrum], settings: PeakSettings) -> None:
    for s in spectra:
        compute_peaks(s, settings)


def snap_index(sp: Spectrum, x_click: float, half_window: float) -> int:
    """Index of the highest point within +-half_window cm-1 of x_click. When that point sits on
    a slope at the window's edge, follow the slope up to the peak top (up to 3x the window away)."""
    x, y = sp.x, sp.y
    lo = int(np.searchsorted(x, x_click - half_window))
    hi = int(np.searchsorted(x, x_click + half_window, side="right"))
    lo, hi = max(0, min(lo, len(x) - 1)), max(lo + 1, min(hi, len(x)))
    i = lo + int(np.argmax(y[lo:hi]))
    reach = 3 * half_window
    while 0 < i < len(y) - 1:
        j = i + 1 if y[i + 1] > y[i] else i - 1 if y[i - 1] > y[i] else i
        if j == i or abs(x[j] - x_click) > reach:
            break
        i = j
    return i


def add_peak_near(sp: Spectrum, x_click: float, half_window: float, settings: PeakSettings) -> Peak:
    """Snap to the highest point within +-half_window cm-1 of x_click and keep it as a manual peak."""
    i = snap_index(sp, x_click, half_window)
    xv = float(sp.x[i])
    sp.excluded = [e for e in sp.excluded if not _near(e, xv)]
    if not any(_near(m, xv) for m in sp.manual):
        sp.manual.append(xv)
    compute_peaks(sp, settings)
    return next(p for p in sp.peaks if p.i == i)


def remove_peak(sp: Spectrum, peak: Peak, settings: PeakSettings) -> None:
    if peak.source == "manual":
        sp.manual = [m for m in sp.manual if nearest_index(sp.x, m) != peak.i]
    else:
        sp.excluded.append(float(sp.x[peak.i]))
    compute_peaks(sp, settings)


def clear_manual_edits(spectra: list[Spectrum], settings: PeakSettings) -> None:
    for s in spectra:
        s.manual, s.excluded = [], []
    compute_all_peaks(spectra, settings)


def peak_table(spectra: list[Spectrum]):
    """All peaks as a pandas DataFrame."""
    import pandas as pd
    rows = [{
        "Spectrum": s.name,
        "Raman shift (cm-1)": round(p.x, 2),
        "Intensity": round(p.y, 3),
        "Prominence": round(p.prominence, 3),
        "FWHM (cm-1)": None if p.fwhm is None else round(p.fwhm, 2),
        "Source": p.source,
    } for s in spectra for p in s.peaks]
    return pd.DataFrame(rows, columns=["Spectrum", "Raman shift (cm-1)", "Intensity", "Prominence", "FWHM (cm-1)", "Source"])


# --------------------------------------------------------------------------- layout

def y_at(sp: Spectrum, xv: float) -> float:
    return float(np.interp(xv, sp.x, sp.y))


def offsets(spectra: list[Spectrum], opts: PlotOptions) -> dict[int, float]:
    """Vertical offset per visible spectrum (keyed by id). In a stack the first-listed spectrum
    is on top and every baseline (minimum) sits on its own level."""
    vis = [s for s in spectra if s.shown]
    if opts.layout != "stacked":
        return {id(s): 0.0 for s in vis}
    step = max([s.yrange for s in vis] + [1.0]) * opts.offset_pct / 100
    n = len(vis)
    return {id(s): (n - 1 - k) * step - s.ymin for k, s in enumerate(vis)}


def data_bounds(spectra, opts, x0=None, x1=None):
    offs = offsets(spectra, opts)
    bx0 = bx1 = by0 = by1 = None
    for s in spectra:
        if not s.shown:
            continue
        m = np.ones_like(s.x, dtype=bool)
        if x0 is not None:
            m &= (s.x >= x0) & (s.x <= x1)
        if not m.any():
            continue
        yy = s.y[m] + offs[id(s)]
        lo, hi = float(yy.min()), float(yy.max())
        by0 = lo if by0 is None else min(by0, lo)
        by1 = hi if by1 is None else max(by1, hi)
        bx0 = float(s.x[0]) if bx0 is None else min(bx0, float(s.x[0]))
        bx1 = float(s.x[-1]) if bx1 is None else max(bx1, float(s.x[-1]))
    if by0 is None:
        return None
    return bx0, bx1, by0, by1


def _pad_y(y0, y1, labels):
    r = (y1 - y0) or 1.0
    return y0 - r * 0.04, y1 + r * (0.2 if labels else 0.06)


def fit_view(spectra, opts, x_range=None):
    """(x0, x1, y0, y1) showing all data, or all data within x_range."""
    if x_range is None:
        b = data_bounds(spectra, opts)
        if b is None:
            return (0.0, 3500.0, 0.0, 1.0)
        return (b[0], b[1], *_pad_y(b[2], b[3], opts.labels))
    b = data_bounds(spectra, opts, *x_range)
    if b is None:
        return (*x_range, 0.0, 1.0)
    return (x_range[0], x_range[1], *_pad_y(b[2], b[3], opts.labels))


# --------------------------------------------------------------------------- styles
# All sizes are in points.

SCREEN_STYLE = dict(
    bg="#FFFFFF", ink="#172131", axis_color="#7A8597", grid_color="#E9ECF1",
    tick=9, title=10, title_weight="normal", label=8, name=9, name_weight="bold", legend=9, gtitle=12,
    line=1.1, axis=0.8, tick_dir="out", tick_len=3.5, minor=False, mirror=False, grid=True,
    y_values=True, y_values_stacked=True, legend_loc="upper left", legend_frame=True,
    mk=dict(gap=3.0, h=5.5, hw=3.0, lw=0.9), label_pad=3.0, halo=2.5, xbins=10, ybins=7,
    y_title_stacked="Intensity (a.u., offset)",
)

PRINT_STYLE = dict(
    bg="#FFFFFF", ink="#000000", axis_color="#000000", grid_color="#DDDDDD",
    tick=7.5, title=8.5, title_weight="normal", label=6.5, name=7, name_weight="normal", legend=7, gtitle=9,
    line=0.75, axis=0.6, tick_dir="in", tick_len=3.2, minor=True, mirror=True, grid=False,
    y_values=True, y_values_stacked=False, legend_loc="upper right", legend_frame=False,
    mk=dict(gap=1.6, h=3.4, hw=1.9, lw=0.5), label_pad=1.6, halo=1.6, xbins=6, ybins=5,
    y_title_stacked="Intensity (a.u.)",
)

_SCALED = ("tick", "title", "label", "name", "legend", "gtitle", "line", "axis", "tick_len", "label_pad", "halo")


def scaled_style(style: dict, k: float) -> dict:
    st = dict(style)
    for key in _SCALED:
        st[key] = style[key] * k
    st["mk"] = {n: v * k for n, v in style["mk"].items()}
    return st


def text_len_pt(s: str, size: float) -> float:
    """Advance width of a number label in Arial (digits 0.556 em, point 0.278 em)."""
    return sum(0.278 if c in ".," else 0.556 for c in s) * size


# --------------------------------------------------------------------------- drawing

def draw(ax, spectra: list[Spectrum], opts: PlotOptions, view, style: dict, empty_message: bool = True) -> None:
    """Draw the whole plot into ax for the given view (x0, x1, y0, y1)."""
    st, fig = style, ax.figure
    ax.clear()
    x0, x1, y0, y1 = view
    # x0 may be greater than x1 when the user intentionally reverses the Raman
    # shift axis. Ordered bounds are used only for data/peak selection.
    x_low, x_high = min(x0, x1), max(x0, x1)
    stacked = opts.layout == "stacked"
    vis = [s for s in spectra if s.shown]
    offs = offsets(spectra, opts)

    ax.set_facecolor(st["bg"])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    for spine in ax.spines.values():
        spine.set_linewidth(st["axis"])
        spine.set_color(st["axis_color"])
    ax.tick_params(which="both", direction=st["tick_dir"], top=st["mirror"], right=st["mirror"],
                   width=st["axis"], color=st["axis_color"], labelsize=st["tick"], labelcolor=st["ink"])
    ax.tick_params(which="major", length=st["tick_len"])
    ax.tick_params(which="minor", length=st["tick_len"] * 0.55)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=st["xbins"], steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=st["ybins"], steps=[1, 2, 2.5, 5, 10]))
    if st["minor"]:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    if st["y_values_stacked" if stacked else "y_values"]:
        yfmt = ScalarFormatter(useOffset=False)
        yfmt.set_scientific(False)
        ax.yaxis.set_major_formatter(yfmt)
    else:
        ax.yaxis.set_major_formatter(NullFormatter())
    if st["grid"]:
        ax.grid(True, which="major", color=st["grid_color"], linewidth=0.6)
        ax.set_axisbelow(True)

    if opts.show_axis_titles:
        tfont = dict(fontsize=st["title"], color=st["ink"], fontweight=st["title_weight"])
        ax.set_xlabel("Raman shift (cm$^{-1}$)", labelpad=st["title"] * 0.3, **tfont)
        ax.set_ylabel(st["y_title_stacked"] if stacked else "Intensity (a.u.)", labelpad=st["title"] * 0.35, **tfont)
    if opts.title and opts.show_title:
        ax.set_title(opts.title, fontsize=st["gtitle"], fontweight="bold", color=st["ink"], pad=st["gtitle"] * 0.6)

    if not vis:
        if empty_message:
            ax.text(0.5, 0.5, "Load .dpt files to begin", transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color=st["axis_color"])
        return

    halo = [pe.withStroke(linewidth=st["halo"], foreground=st["bg"])]
    n = len(vis)
    # Curves: the first-listed spectrum is drawn on top.
    for k, s in enumerate(vis):
        ax.plot(s.x, s.y + offs[id(s)], color=s.color, lw=st["line"], label=s.name,
                zorder=2 + (n - k) * 0.01, solid_joinstyle="round")

    # Peak markers: triangles sitting just above each peak.
    mk = st["mk"]
    mark_tf = transforms.offset_copy(ax.transData, fig=fig, x=0, y=mk["gap"] + mk["h"] / 2, units="points")
    to_pt = 72.0 / fig.dpi
    placed = []
    for k, s in enumerate(vis):
        o = offs[id(s)]
        ps = [p for p in s.peaks if x_low <= p.x <= x_high]
        for src, face in (("auto", s.color), ("manual", st["bg"])):
            sel = [p for p in ps if p.source == src]
            if sel:
                ax.plot([p.x for p in sel], [p.y + o for p in sel], ls="none", marker="v",
                        ms=mk["hw"] * 2, mfc=face, mec=s.color, mew=mk["lw"], transform=mark_tf,
                        zorder=3 + (n - k) * 0.01, clip_on=True)
        for p in ps:
            if y0 <= p.y + o <= y1:
                placed.append((s, p, o))

    # Wavenumber labels, most prominent first; a label that would overlap one already placed is skipped.
    if opts.labels:
        placed.sort(key=lambda t: -t[1].prominence / t[0].yrange)
        boxes = []
        half = st["label"] * 0.62
        base = mk["gap"] + mk["h"] + st["label_pad"]
        for s, p, o in placed:
            X, Y = ax.transData.transform((p.x, p.y + o)) * to_pt
            text = f"{p.x:.1f}"
            ln = text_len_pt(text, st["label"])
            box = (X - half, X + half, Y + base, Y + base + ln + 1)
            if any(b[0] < box[1] and box[0] < b[1] and b[2] < box[3] and box[2] < b[3] for b in boxes):
                continue
            boxes.append(box)
            t = ax.annotate(text, xy=(p.x, p.y + o), xytext=(0, base), textcoords="offset points",
                            rotation=90, ha="center", va="bottom", fontsize=st["label"], color=s.color,
                            path_effects=halo, annotation_clip=True, zorder=5)
            t.set_in_layout(False)

    if opts.show_names:
        if stacked:
            blend = transforms.blended_transform_factory(ax.transAxes, ax.transData)
            for s in vis:
                # Use a data position just inside the visual right edge. This
                # works for both normal and reversed Raman-shift axes.
                visual_right = x1
                inward_direction = 1.0 if x1 < x0 else -1.0
                xr = visual_right + inward_direction * abs(x1 - x0) * 0.01
                xr = float(np.clip(xr, s.x[0], s.x[-1]))
                t = ax.annotate(s.name, xy=(1, y_at(s, xr) + offs[id(s)]), xycoords=blend,
                                xytext=(-st["name"] * 0.7, st["name"] * 0.45), textcoords="offset points",
                                ha="right", va="bottom", fontsize=st["name"], fontweight=st["name_weight"],
                                color=s.color, path_effects=halo, annotation_clip=False, clip_on=True, zorder=6)
                t.set_in_layout(False)
        else:
            leg = ax.legend(loc=st["legend_loc"], frameon=st["legend_frame"], fontsize=st["legend"],
                            handlelength=1.8, borderaxespad=0.7, labelcolor=st["ink"])
            if st["legend_frame"]:
                leg.get_frame().set_edgecolor(st["grid_color"])
                leg.get_frame().set_linewidth(0.8)
                leg.get_frame().set_alpha(1)


def label_safe_top(ax, spectra, opts, view, style) -> float:
    """Top of the intensity range needed so every peak whose point is inside view keeps its
    marker and label inside the axes. Peaks above or beside the view are ignored, so a
    deliberate zoom into small peaks is never undone."""
    x0, x1, y0, y1 = view
    x_low, x_high = min(x0, x1), max(x0, x1)
    h_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
    if h_pt <= 0:
        return y1
    mk = style["mk"]
    base = mk["gap"] + mk["h"] + 2
    offs = offsets(spectra, opts)
    top = y1
    for s in spectra:
        if not s.shown:
            continue
        o = offs[id(s)]
        for p in s.peaks:
            yp = p.y + o
            if not (x_low <= p.x <= x_high and y0 <= yp <= y1):
                continue
            need = base + (style["label_pad"] + text_len_pt(f"{p.x:.1f}", style["label"]) if opts.labels else 0)
            r = need / h_pt
            if r < 0.8:
                top = max(top, y0 + (yp - y0) / (1 - r))
    return top


# --------------------------------------------------------------------------- export

def journal_figure(spectra, opts, view, width_cm: float = 8.5, aspect: float = 0.75, scale: float = 1.0):
    """A publication figure at its true physical size (Arial, inward ticks on all sides, no grid)."""
    w_in = width_cm / 2.54
    fig = Figure(figsize=(w_in, w_in * aspect), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    st = scaled_style(PRINT_STYLE, scale)
    draw(ax, spectra, opts, view, st, empty_message=False)
    fig.tight_layout(pad=0.3)
    top = label_safe_top(ax, spectra, opts, view, st)
    if top > view[3]:
        view = (view[0], view[1], view[2], top)
        draw(ax, spectra, opts, view, st, empty_message=False)
        fig.tight_layout(pad=0.3)
    return fig


def export_figure(path, spectra, opts, view, width_cm=8.5, aspect=0.75, scale=1.0, dpi=600) -> Path:
    """Save as .png, .tif, .svg or .pdf (chosen by the file extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = journal_figure(spectra, opts, view, width_cm, aspect, scale)
    ext = path.suffix.lower()
    kw = dict(facecolor="white")
    if ext in (".png", ".tif", ".tiff"):
        kw["dpi"] = dpi  # stored in the file, so Word/LaTeX place it at the true width
    if ext in (".tif", ".tiff"):
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(path, **kw)
    return path


def safe_filename(text: str, fallback: str = "raman-figure") -> str:
    s = re.sub(r"[^\w\- ]+", "", text).strip()
    return re.sub(r"\s+", "-", s)[:60] or fallback
