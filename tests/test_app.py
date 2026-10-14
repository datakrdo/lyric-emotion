"""Smoke test: every page of the dashboard renders from the committed bundle."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.skipif(not (APP.parent / "data" / "findings.json").exists(), reason="no bundle")
def test_landing_page_renders_verdict_and_trust_panel_without_error():
    at = AppTest.from_file(str(APP)).run(timeout=120)
    assert not at.exception
    assert [h.value for h in at.subheader] == ["How sure are we?", "What each band shows more of"]
    assert at.get("plotly_chart")  # the two-axis map

