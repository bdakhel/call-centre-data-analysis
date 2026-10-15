"""The DAX and M files must match the tables build_model_tables.py produces.

A mistyped column name in either file does not fail here — it fails later,
inside Power BI, as a broken measure or an untyped column. These tests move
that failure forward to the commit that caused it.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

MEASURES = Path("powerbi/measures.dax")
QUERIES = Path("powerbi/load_queries.m")
PROCESSED = Path("data/processed")

TABLE_REF = re.compile(r"\b(dim_\w+|fact_\w+)\[(\w+)\]")


@pytest.fixture(scope="module")
def schema():
    files = sorted(PROCESSED.glob("*.csv"))
    tables = {p.stem: list(pd.read_csv(p, nrows=1).columns)
              for p in files if p.stem.startswith(("dim_", "fact_"))}
    if not tables:
        pytest.skip("model tables missing -- run src/build_model_tables.py")
    return tables


@pytest.fixture(scope="module")
def dax():
    if not MEASURES.exists():
        pytest.skip("powerbi/measures.dax missing")
    # The Tabular Editor C# block at the end is commented out; its strings
    # are not DAX and must not be parsed as such.
    return MEASURES.read_text().split("OPTIONAL — Tabular Editor")[0]


@pytest.fixture(scope="module")
def mcode():
    if not QUERIES.exists():
        pytest.skip("powerbi/load_queries.m missing")
    return QUERIES.read_text()


def measure_names(dax):
    """Measure definitions: a name on its own line ending in '='.

    VAR declarations match the same shape, so they are filtered out -- they
    are locally scoped and must not count as defined measures.
    """
    names = set()
    for line in dax.splitlines():
        m = re.match(r"^([A-Za-z%][^=\n]*?)\s*=\s*$", line)
        if m and not line.strip().startswith(("VAR ", "RETURN")):
            names.add(m.group(1).strip())
    return names


# --- DAX ------------------------------------------------------------------

def test_dax_column_references_exist(dax, schema):
    broken = []
    for table, col in sorted(set(TABLE_REF.findall(dax))):
        if table not in schema:
            broken.append(f"{table}[{col}] (no such table)")
        elif col not in schema[table]:
            broken.append(f"{table}[{col}] (no such column)")
    assert not broken, "broken DAX column references:\n  " + "\n  ".join(broken)


def test_dax_measure_references_are_defined(dax):
    defined = measure_names(dax)
    columns = {c for _, c in TABLE_REF.findall(dax)}
    local_vars = set(re.findall(r"^\s*VAR\s+(\w+)", dax, re.M))
    referenced = set(re.findall(r"(?<![A-Za-z_])\[([A-Z%][^\]]*?)\]", dax))
    undefined = referenced - defined - columns - local_vars
    assert not undefined, f"measures referenced but never defined: {sorted(undefined)}"


def test_required_measures_are_present(dax):
    """The list the model was commissioned with."""
    defined = measure_names(dax)
    required = [
        "CSAT %", "Survey Response Rate %", "FCR % (SQM)",
        "FCR vs Benchmark (pp)", "AHT (minutes)", "AHT vs Target %",
        "Net Sentiment", "% Negative", "Repeat Contact Rate %",
        "Sentiment Shift", "Interactions",
    ]
    assert not [m for m in required if m not in defined], \
        f"missing: {[m for m in required if m not in defined]}"


def test_mom_and_variance_measures_exist(dax):
    defined = measure_names(dax)
    assert len([m for m in defined if "MoM" in m]) >= 6, "too few MoM measures"
    assert len([m for m in defined if "Target" in m or "Benchmark" in m]) >= 5, \
        "too few target/benchmark variance measures"


def test_rate_changes_are_in_percentage_points(dax):
    """The change in a percentage is in points, not percent -- the names must
    say so, or a reader will double-count the scaling."""
    for name in measure_names(dax):
        if "MoM" in name and any(k in name for k in ("CSAT", "Negative", "Repeat", "FCR")):
            assert "(pp)" in name, f"{name} should be suffixed (pp)"


def test_no_bare_division(dax):
    """'/' errors on a zero denominator; DIVIDE() returns blank."""
    offenders = [ln.strip() for ln in dax.splitlines()
                 if re.search(r"\]\s*/\s*[A-Za-z\[]", ln) and not ln.strip().startswith("//")]
    assert not offenders, f"use DIVIDE() instead of '/': {offenders}"


# --- Power Query M --------------------------------------------------------

def m_blocks(mcode):
    return dict(
        (f, re.findall(r'\{"(\w+)",', body))
        for f, body in re.findall(
            r'fnLoadTable\(\s*"(\w+)\.csv",\s*\{(.*?)\n\s*\}\s*\)', mcode, re.S)
    )


def test_m_covers_every_table(mcode, schema):
    assert set(m_blocks(mcode)) == set(schema), \
        f"M loads {sorted(m_blocks(mcode))}, model has {sorted(schema)}"


def test_m_types_every_column_exactly_once(mcode, schema):
    problems = []
    for table, listed in m_blocks(mcode).items():
        actual = schema[table]
        if len(listed) != len(set(listed)):
            problems.append(f"{table}: duplicate column in the type list")
        for col in set(actual) - set(listed):
            problems.append(f"{table}.{col} is in the CSV but never typed")
        for col in set(listed) - set(actual):
            problems.append(f"{table}.{col} is typed but not in the CSV")
    assert not problems, "\n  " + "\n  ".join(problems)


def test_m_sets_utf8_and_csv_quoting(mcode):
    """Both are load-bearing: the text columns contain em-dashes, commas and
    embedded newlines."""
    assert "Encoding = 65001" in mcode, "UTF-8 encoding not pinned"
    assert "QuoteStyle = QuoteStyle.Csv" in mcode, "CSV quoting not pinned"


def test_m_pins_culture_for_date_parsing(mcode):
    """ISO dates parse differently under a dd/MM locale without this."""
    assert '"en-US"' in mcode, "Table.TransformColumnTypes culture not pinned"


def test_m_uses_the_folder_parameter(mcode):
    assert "DataFolder" in mcode
    assert "C:\\path\\to" in mcode, "placeholder path should be obviously a placeholder"
