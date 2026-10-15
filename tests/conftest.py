"""Shared pytest fixtures.

The src/ modules use paths relative to the project root ("data/raw/..."), so
tests chdir there rather than depending on where pytest was invoked from.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session", autouse=True)
def _run_from_project_root():
    """Every module in src/ resolves data paths relative to the repo root."""
    previous = Path.cwd()
    os.chdir(PROJECT_ROOT)
    yield
    os.chdir(previous)


@pytest.fixture(scope="session")
def interactions():
    path = PROJECT_ROOT / "data/raw/interactions.csv"
    if not path.exists():
        pytest.skip("data/raw/interactions.csv missing -- run src/generate_all.py")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def agents():
    path = PROJECT_ROOT / "data/raw/agents.csv"
    if not path.exists():
        pytest.skip("data/raw/agents.csv missing -- run src/generate_all.py")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def sample(interactions):
    """A small slice, big enough to cover several topics and both sentiments."""
    return interactions.head(40).copy()


@pytest.fixture(scope="session")
def vader_scores(sample):
    """Stage 1 output on the sample, via the fast scorer.

    Session-scoped: scoring once and sharing keeps the suite quick.
    """
    import sentiment

    return sentiment.VaderScorer(use_cache=False).score_interactions(sample)


@pytest.fixture(scope="session")
def topic_predictions(sample):
    import topics

    return topics.RuleTopicClassifier().predict(
        sample["transcript_text"].astype(str).tolist(), show_progress=False)


@pytest.fixture(scope="session")
def scores_file():
    """The real pipeline artefact, if it has been produced."""
    path = PROJECT_ROOT / "data/processed/interaction_scores.csv"
    if not path.exists():
        pytest.skip("interaction_scores.csv missing -- run src/run_pipeline.py")
    return pd.read_csv(path)
