import numpy as np

import raman_core as rc


def test_parse_dpt_skips_headers_and_sorts_raman_shift():
    text = "Header line\n300,9\n100,1\n200,5\n"

    x_values, y_values = rc.parse_dpt(text)

    assert np.array_equal(x_values, np.array([100.0, 200.0, 300.0]))
    assert np.array_equal(y_values, np.array([1.0, 5.0, 9.0]))


def test_peak_detection_and_table_generation():
    spectrum = rc.Spectrum(
        name="Sample",
        x=np.arange(7.0),
        y=np.array([0.0, 1.0, 4.0, 1.0, 0.0, 2.0, 0.0]),
    )
    settings = rc.PeakSettings(
        auto=True,
        min_prominence_pct=10.0,
        min_spacing=1.0,
    )

    rc.compute_peaks(spectrum, settings)
    table = rc.peak_table([spectrum])

    assert len(spectrum.peaks) == 2
    assert list(table["Spectrum"]) == ["Sample", "Sample"]
    assert list(table["Raman shift (cm-1)"]) == [2.0, 5.0]
