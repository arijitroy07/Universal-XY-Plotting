import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_UNITS = {
    "%", "a", "a.u.", "au", "arb.", "c", "cm", "count", "counts", "deg",
    "degree", "degrees", "ev", "gpa", "h", "hr", "hz", "k", "kev", "khz",
    "kn", "kohm", "kv", "m", "ma", "mev", "mhz", "min", "mm", "mn",
    "mohm", "mpa", "ms", "mv", "mw", "n", "na", "nm", "ns", "ohm",
    "pa", "s", "ua", "um", "us", "v", "w", "°c", "µa", "µm", "µs",
    "μa", "μm", "μs", "ω",
}


def cell_to_string(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def clean_numeric_series(series):
    """Convert text-like values to numbers while discarding other content."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("−", "-", regex=False)
        .str.replace("–", "-", regex=False)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def numeric_view(df):
    return pd.DataFrame(
        {column: clean_numeric_series(df[column]) for column in df.columns},
        index=df.index,
    )


def score_candidate_dataframe(df):
    if df is None or df.empty:
        return -1
    numeric = numeric_view(df)
    numeric_per_row = numeric.notna().sum(axis=1)
    rows_with_two_numbers = int((numeric_per_row >= 2).sum())
    if rows_with_two_numbers == 0:
        return 0
    median_columns = float(numeric_per_row[numeric_per_row >= 2].median())
    return rows_with_two_numbers * 10 + median_columns + min(df.shape[1], 20) * 0.01


def read_text_table(file_bytes):
    text = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = file_bytes.decode("latin-1", errors="ignore")

    candidates = []
    for separator in (None, "\t", ",", ";", r"\s+"):
        try:
            frame = pd.read_csv(
                io.StringIO(text),
                sep=separator,
                engine="python",
                header=None,
                dtype=str,
                skip_blank_lines=False,
                on_bad_lines="skip",
            ).replace(r"^\s*$", np.nan, regex=True)
            candidates.append((score_candidate_dataframe(frame), frame, separator))
        except Exception:
            continue
    if not candidates:
        raise ValueError("The text file could not be interpreted.")
    return max(candidates, key=lambda item: item[0])[1:]


def looks_like_unit_token(value):
    text = cell_to_string(value)
    if not text:
        return False
    lowered = text.lower().strip()
    if (lowered.startswith("(") and lowered.endswith(")")) or (
        lowered.startswith("[") and lowered.endswith("]")
    ):
        return True
    if lowered.strip("()[] ") in COMMON_UNITS:
        return True
    patterns = (
        r".*/.*", r".*\^-?\d+.*", r".*[a-zA-Z]+-?\d+$", r".*cm-1.*",
        r".*cm\^-1.*", r".*g-1.*", r".*g\^-1.*", r".*mAh.*", r".*Wh.*",
        r".*mol.*",
    )
    return len(text) <= 25 and any(re.match(p, text, re.IGNORECASE) for p in patterns)


def looks_like_unit_row(raw_df, row_index, active_columns):
    if row_index < 0 or row_index >= len(raw_df):
        return False
    values = [
        cell_to_string(raw_df.loc[row_index, column]) for column in active_columns
    ]
    values = [value for value in values if value]
    return bool(values) and sum(map(looks_like_unit_token, values)) / len(values) >= 0.5


def detect_data_start(raw_df):
    counts = numeric_view(raw_df).notna().sum(axis=1)
    for position in range(len(raw_df)):
        if counts.iloc[position] < 2:
            continue
        window = counts.iloc[position : min(position + 6, len(raw_df))]
        if (window >= 2).sum() >= min(3, len(window)):
            return position
    candidates = np.where(counts.to_numpy() >= 2)[0]
    return int(candidates[0]) if len(candidates) else 0


def determine_active_columns(raw_df, data_start):
    section = numeric_view(raw_df).iloc[data_start : min(data_start + 200, len(raw_df))]
    counts = section.notna().sum(axis=0)
    minimum_points = 1 if len(section) <= 2 else 2
    active = [column for column in raw_df.columns if counts[column] >= minimum_points]
    if len(active) < 2:
        active = list(counts.sort_values(ascending=False).index[: min(2, len(counts))])
    return active


def row_text_score(raw_df, row_index, active_columns):
    if row_index < 0 or row_index >= len(raw_df):
        return -1
    return sum(
        bool(value) and pd.isna(pd.to_numeric(value, errors="coerce"))
        for value in (cell_to_string(raw_df.loc[row_index, col]) for col in active_columns)
    )


def get_header_and_unit_rows(raw_df, data_start, active_columns):
    if data_start == 0:
        return None, None
    previous_row = data_start - 1
    unit_row = (
        previous_row
        if looks_like_unit_row(raw_df, previous_row, active_columns)
        else None
    )
    search_end = unit_row - 1 if unit_row is not None else previous_row
    search_start = max(0, search_end - 5)
    candidates = [
        (row_text_score(raw_df, row, active_columns) * 100 + row, row)
        for row in range(search_start, search_end + 1)
    ]
    best_score, header_row = max(candidates, default=(-1, None))
    return (header_row if best_score > 0 else None), unit_row


def strip_unit_brackets(value):
    text = cell_to_string(value).strip()
    if len(text) >= 2 and ((text[0], text[-1]) in {("(", ")"), ("[", "]")}):
        text = text[1:-1]
    return text.strip()


def make_unique_names(names):
    counts = {}
    output = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        output.append(name if counts[name] == 1 else f"{name} [{counts[name]}]")
    return output


def prepare_table(
    raw_df,
    data_start_override=None,
    header_row_override=None,
    unit_row_override=None,
):
    raw_df = (
        raw_df.copy()
        .replace(r"^\s*$", np.nan, regex=True)
        .dropna(how="all")
        .dropna(axis=1, how="all")
        .reset_index(drop=True)
    )
    if raw_df.empty:
        raise ValueError("The file does not contain a table.")
    raw_df.columns = range(raw_df.shape[1])

    data_start = (
        data_start_override
        if data_start_override is not None
        else detect_data_start(raw_df)
    )
    if data_start < 0 or data_start >= len(raw_df):
        raise ValueError("The selected first data row is outside the file.")
    active_columns = determine_active_columns(raw_df, data_start)

    if data_start_override is None:
        header_row, unit_row = get_header_and_unit_rows(
            raw_df, data_start, active_columns
        )
    else:
        header_row, unit_row = header_row_override, unit_row_override

    names = []
    for position, column in enumerate(active_columns):
        header = (
            cell_to_string(raw_df.loc[header_row, column])
            if header_row is not None and 0 <= header_row < len(raw_df)
            else ""
        )
        header = header or f"Column {position + 1}"
        unit = (
            strip_unit_brackets(raw_df.loc[unit_row, column])
            if unit_row is not None and 0 <= unit_row < len(raw_df)
            else ""
        )
        names.append(
            f"{header} ({unit})" if unit and unit.lower() not in header.lower() else header
        )

    prepared = raw_df.loc[data_start:, active_columns].copy().reset_index(drop=True)
    prepared.columns = make_unique_names(names)
    metadata = {
        "data_start": data_start,
        "header_row": header_row,
        "unit_row": unit_row,
        "active_columns": active_columns,
    }
    return prepared, metadata


def get_excel_sheets(file_bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def read_uploaded_file(uploaded_file, sheet_name=None):
    extension = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    if extension in {".xls", ".xlsx"}:
        frame = pd.read_excel(
            io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, dtype=object
        )
        return frame, None
    return read_text_table(file_bytes)
