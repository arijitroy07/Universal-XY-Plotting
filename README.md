# Universal XY and Raman Data Plotters

An interactive multipage Streamlit application for plotting general experimental
XY data and analysing Raman spectra.

## Features

### Universal XY Data Plotter

- Upload one or many files.
- Select the X and Y columns independently for every file.
- Automatically locate numeric data and discard text or metadata rows.
- Detect headers and units written in a header or in a separate units row.
- Manually override the first data row, header row, and units row when needed.
- Plot datasets as lines, scatter points, or scatter points with lines.
- Display overlapping traces, independently scaled and vertically offset
  traces, or stacked panels.
- Customize figure dimensions, labels, title, legend, axis limits, tick spacing,
  grid, line width, marker size, and trace colors.
- Create or update the plot only when the plot button is selected.
- Download the generated plot as a single high-quality JPEG.

### Raman Data Plotter

- Upload multiple DPT, TXT, CSV, DAT, or PRN Raman spectra.
- Rename, recolor, reorder, show, or hide individual spectra.
- Display spectra as overlaid traces or vertically stacked traces.
- Crop the Raman-shift data before plotting and peak detection.
- Detect peaks using adjustable prominence and spacing criteria.
- Add manual peak positions or exclude unwanted automatic peaks.
- Display peak position, intensity, prominence, FWHM, and source in a table.
- Customize titles, labels, stacking distance, axis ranges, figure width,
  aspect ratio, and raster resolution.
- Reverse the Raman-shift axis by entering an X minimum greater than X maximum.
- Automatically fit the Y range to data visible inside a customized X range.
- Create or update the figure only when the plot button is selected.
- Download publication-ready PNG, TIFF, SVG, and PDF figures and a peak-table CSV.

Matplotlib math notation can be used in titles, axis labels, and legend labels.
For example, enter `Capacity (mAh g$^{-1}$)` to display the `-1` as a
superscript, or `MoS$_2$` to display the `2` as a subscript.

The X-axis can be reversed by entering an X minimum that is greater than the X
maximum. For example, use an X minimum of `100` and an X maximum of `0` to show
values decreasing from left to right.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bat
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies and launch the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The `pip` and `streamlit` commands belong in Command Prompt, PowerShell,
Terminal, or Anaconda Prompt. Do not paste them into `app.py` or run them as
Python statements in Spyder.

## Upload to GitHub

1. Create a new empty repository on GitHub.
2. Extract this project and upload the files and folders at the repository root.
3. Confirm that `app.py`, `raman_core.py`, `requirements.txt`, `runtime.txt`,
   and the `pages` folder are present in the repository.
4. Commit the files to the default branch.

You can also push the project with Git:

```bash
git init
git add .
git commit -m "Initial Streamlit XY plotter"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/).
2. Select **Create app**.
3. Choose this GitHub repository and its `main` branch.
4. Enter `app.py` as the main file path.
5. Select **Deploy**.

No secrets or environment variables are required.

When these files are committed to the same GitHub repository used by an existing
Streamlit Community Cloud deployment, the deployment link remains unchanged.
Streamlit redeploys the repository and adds **Raman Data Plotter** to the page
navigation automatically.

## Repository structure

```text
universal-xy-plotter/
├── .github/
│   └── workflows/
│       └── tests.yml
├── .streamlit/
│   └── config.toml
├── .gitignore
├── app.py
├── data_reader.py
├── plotting.py
├── raman_core.py
├── requirements.txt
├── runtime.txt
├── pages/
│   └── 2_Raman_Data_Plotter.py
├── tests/
│   ├── test_app.py
│   ├── test_data_reader.py
│   ├── test_raman_core.py
│   └── test_plotting.py
└── README.md
```

## Notes about input files

The automatic reader works best when the main data table contains at least two
numeric columns and several numeric rows. If an instrument export contains
numeric metadata before the real table, open **Detection settings and raw
preview**, select **Manual override**, and enter the correct row numbers.

In **Stacked traces with Y offset** mode, every trace is automatically scaled
from its own minimum to maximum before the vertical offset is added. The Y-axis
label remains visible, but the Y-axis numbers are hidden.

Raman text files must contain at least two numeric columns. The first numeric
column is interpreted as Raman shift and the second as intensity. Header,
comment, and other nonnumeric lines are ignored automatically.
