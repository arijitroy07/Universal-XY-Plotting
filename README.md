# Universal XY Plotter

An interactive Streamlit application for plotting experimental XY data from
XLS, XLSX, CSV, TXT, and XY files.

## Features

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

Matplotlib math notation can be used in titles, axis labels, and legend labels.
For example, enter `Capacity (mAh g$^{-1}$)` to display the `-1` as a
superscript, or `MoS$_2$` to display the `2` as a subscript.

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
3. Confirm that `app.py`, `requirements.txt`, and `runtime.txt` are visible at
   the top level of the repository.
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
├── requirements.txt
├── runtime.txt
├── tests/
│   ├── test_app.py
│   ├── test_data_reader.py
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
