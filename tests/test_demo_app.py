"""The demo app's non-UI logic.

The accuracy parser is the piece worth testing: it reads docs/validation.md so
the sidebar cannot quote figures the validation no longer supports. If someone
rewords that table, the app silently loses its numbers — this catches it at
commit time instead.
"""

import pytest

demo = pytest.importorskip("demo_app", reason="streamlit not installed")


@pytest.fixture(scope="module")
def accuracy():
    # .__wrapped__ bypasses st.cache_data, which needs a script run context.
    return demo.load_accuracy.__wrapped__()


def test_accuracy_parses_from_validation_doc(accuracy):
    assert accuracy, "nothing parsed — docs/validation.md missing or reworded"


@pytest.mark.parametrize("key", ["accuracy", "kappa", "ceiling", "whole_call", "end_only", "n"])
def test_every_sidebar_figure_is_present(accuracy, key):
    assert key in accuracy, f"'{key}' no longer parses out of docs/validation.md"


def test_parsed_figures_are_plausible(accuracy):
    assert 0 < accuracy["accuracy"] <= 100
    assert 0 <= accuracy["kappa"] <= 1
    assert accuracy["ceiling"] > accuracy["accuracy"], \
        "the ceiling must beat the model, or the framing is wrong"
    assert accuracy["end_only"] > accuracy["whole_call"], \
        "the end-of-call lift is the app's headline claim"


def test_parser_degrades_quietly_rather_than_lying(tmp_path, monkeypatch):
    """A missing file must yield no figures, never stale ones."""
    monkeypatch.setattr(demo, "VALIDATION", tmp_path / "gone.md")
    assert demo.load_accuracy.__wrapped__() == {}


def test_parser_survives_a_reworded_file(tmp_path, monkeypatch):
    """Partial matches are fine; a crash is not."""
    doc = tmp_path / "validation.md"
    doc.write_text("# Nothing this parser recognises\n\nsome prose\n")
    monkeypatch.setattr(demo, "VALIDATION", doc)
    assert demo.load_accuracy.__wrapped__() == {}


# --- Verdict bands --------------------------------------------------------

@pytest.mark.parametrize("end_score,expected", [
    (-1.0, "negative"), (-0.5, "negative"), (-0.26, "negative"),
    (-0.25, "neutral"), (0.0, "neutral"), (0.89, "neutral"),
    (0.95, "positive"), (1.0, "positive"),
])
def test_verdict_uses_the_validated_band(end_score, expected):
    assert demo.verdict(end_score)[0] == expected


def test_verdict_bands_match_the_documented_thresholds():
    """The bands live in measures.py now — one definition shared by the report
    pages and the live scorer, rather than a copy in each."""
    import measures

    assert (measures.NEG_BAND, measures.POS_BAND) == (-0.25, 0.90)
    # And the scorer's verdict() must actually use them.
    assert demo.verdict(measures.NEG_BAND - 0.01)[0] == "negative"
    assert demo.verdict(measures.POS_BAND + 0.01)[0] == "positive"


def test_every_verdict_has_a_colour():
    for score in (-1, 0, 1):
        label, colour = demo.verdict(score)
        assert colour.startswith("#") and len(colour) == 7


# --- Samples --------------------------------------------------------------

def test_samples_cover_every_topic(accuracy):
    samples = demo.load_samples.__wrapped__()
    if samples.empty:
        pytest.skip("no source data")
    import topics
    assert set(samples["topic"]) == set(topics.TOPICS)
    assert samples["transcript_text"].str.len().gt(100).all()


# --- Streamlit API compatibility ------------------------------------------
# The app is run from whichever Python the reader has, not necessarily the
# project venv. Streamlit 1.6x renamed use_container_width to width="stretch",
# but 1.37 raises TypeError on `width` for altair_chart and button — so the app
# must stick to the argument that works on both.

def test_app_uses_only_width_kwargs_the_installed_streamlit_accepts():
    import inspect
    import re
    from pathlib import Path

    import streamlit as st

    source = Path("src/demo_app.py").read_text()
    # Strip comments so the explanatory note about the rename is not read as code.
    code = "\n".join(l for l in source.splitlines() if not l.lstrip().startswith("#"))

    for fn_name, fn in [("altair_chart", st.altair_chart), ("dataframe", st.dataframe),
                        ("button", st.button)]:
        params = inspect.signature(fn).parameters
        calls = re.findall(rf"st\.{fn_name}\((?:[^()]|\([^()]*\))*\)", code, re.S)
        for call in calls:
            if 'width="stretch"' in call and "width" not in params:
                raise AssertionError(
                    f'st.{fn_name} is called with width="stretch" but Streamlit '
                    f"{st.__version__} does not accept it")
            if "use_container_width" in call and "use_container_width" not in params:
                raise AssertionError(
                    f"st.{fn_name} is called with use_container_width but Streamlit "
                    f"{st.__version__} has removed it")


def test_scorer_reports_missing_model_dependencies():
    """The report pages need only pandas/altair; scoring needs the models.
    A Python without them must say so, not throw ImportError on click."""
    missing = demo.scorer_available()
    assert isinstance(missing, list)
    for name in missing:
        assert name in ("torch", "transformers", "vaderSentiment")
