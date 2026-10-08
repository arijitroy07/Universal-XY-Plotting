from pathlib import Path

from streamlit.testing.v1 import AppTest


SAMPLE_RAMAN = b"Raman shift,Intensity\n100,1\n200,5\n300,1\n400,6\n500,1\n"


def test_raman_page_waits_for_plot_button_then_creates_outputs():
    page_path = (
        Path(__file__).resolve().parents[1]
        / "pages"
        / "2_Raman_Data_Plotter.py"
    )
    app = AppTest.from_file(page_path).run(timeout=30)

    app.get("file_uploader")[0].upload(
        "sample.dpt",
        SAMPLE_RAMAN,
        "text/plain",
    ).run(timeout=30)

    assert not app.exception
    assert len(app.get("image")) == 0
    assert len(app.get("download_button")) == 0

    app.button[0].click().run(timeout=60)

    assert not app.exception
    assert len(app.get("image")) == 1
    assert len(app.get("download_button")) >= 4
