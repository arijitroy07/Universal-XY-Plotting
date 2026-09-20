from pathlib import Path

from streamlit.testing.v1 import AppTest


SAMPLE_FILE = b"X,Y\n(s),(mA)\n0,10\n1,20\n2,15\n"


def test_plot_is_created_only_after_button_click_and_has_one_download():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=20)
    app.get("file_uploader")[0].upload(
        "sample.csv", SAMPLE_FILE, "text/csv"
    ).run(timeout=20)

    assert not app.exception
    assert len(app.get("image")) == 0
    assert len(app.get("download_button")) == 0

    app.button[0].click().run(timeout=20)

    assert not app.exception
    assert len(app.get("image")) == 1
    assert len(app.get("download_button")) == 1
    assert app.get("download_button")[0].proto.label == "Download JPEG"
    assert app.get("download_button")[0].proto.url.endswith(".jpg")
