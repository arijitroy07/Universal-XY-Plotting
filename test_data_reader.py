import pandas as pd

from data_reader import prepare_table


def test_detects_header_and_separate_units_row():
    raw = pd.DataFrame(
        [
            ["Instrument export", None],
            ["Potential", "Current"],
            ["(V)", "(mA)"],
            [0.0, 1.0],
            [0.1, 1.5],
            [0.2, 2.0],
        ]
    )
    prepared, metadata = prepare_table(raw)
    assert list(prepared.columns) == ["Potential (V)", "Current (mA)"]
    assert metadata["data_start"] == 3


def test_duplicate_headers_are_made_unique():
    raw = pd.DataFrame(
        [["Time", "Signal", "Signal"], [0, 1, 2], [1, 2, 3], [2, 3, 4]]
    )
    prepared, _ = prepare_table(raw)
    assert list(prepared.columns) == ["Time", "Signal", "Signal [2]"]
