"""Run every data generator in order: data/raw/ from empty to complete.

Order matters -- each script reads what the previous one wrote:

    1. generate_agents.py        -> agents.csv
    2. generate_interactions.py  -> interactions.csv (needs agents.csv)
    3. generate_transcripts.py   -> fills transcript_text in interactions.csv
    4. generate_cases.py         -> cases.csv        (needs interactions.csv)
    5. generate_surveys.py       -> csat_survey.csv, sqm_monthly.csv
                                    (needs interactions.csv + cases.csv)

Each generator runs as its own SUBPROCESS rather than being imported and
called. That is deliberate: every script seeds `random` / `numpy` at import
time, so importing them all into one process would leave the seeding order
tangled and the output would no longer match a standalone run. A fresh
interpreter per script keeps the whole pipeline byte-for-byte reproducible.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/generate_all.py            # run everything
    python src/generate_all.py --quiet    # only show each step's result
"""

import subprocess
import sys
import time
from pathlib import Path

# Run order. Each entry is (script, files it is expected to produce).
PIPELINE = [
    ("src/generate_agents.py", ["data/raw/agents.csv"]),
    ("src/generate_interactions.py", ["data/raw/interactions.csv"]),
    ("src/generate_transcripts.py", ["data/raw/interactions.csv"]),
    ("src/generate_cases.py", ["data/raw/cases.csv"]),
    ("src/generate_surveys.py", ["data/raw/csat_survey.csv", "data/raw/sqm_monthly.csv"]),
]


def run_step(script, quiet):
    """Run one generator with the SAME interpreter that launched this script.

    Using sys.executable (not a bare "python") means that if you ran this from
    a virtualenv, the generators get that venv's pandas/faker too.
    """
    started = time.perf_counter()
    # Output is streamed straight through unless --quiet, so you still see each
    # generator's own summary and pattern checks.
    result = subprocess.run(
        [sys.executable, script],
        capture_output=quiet,
        text=True,
    )
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        if quiet and result.stdout:
            print(result.stdout)
        if quiet and result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(f"\n{script} failed (exit {result.returncode}) -- pipeline stopped.")

    return elapsed


def main():
    quiet = "--quiet" in sys.argv

    # Guard against being run from inside src/ -- every generator uses paths
    # relative to the project root.
    if not Path("src/generate_agents.py").exists():
        raise SystemExit("Run this from the project root: python src/generate_all.py")

    print(f"Running {len(PIPELINE)} generators with {sys.executable}\n")

    total = 0.0
    for i, (script, outputs) in enumerate(PIPELINE, start=1):
        header = f"[{i}/{len(PIPELINE)}] {script}"
        print(f"{header}\n{'-' * len(header)}")

        elapsed = run_step(script, quiet)
        total += elapsed

        # Confirm the step actually left the files behind that we expect.
        missing = [p for p in outputs if not Path(p).exists()]
        if missing:
            raise SystemExit(f"{script} exited cleanly but did not write: {', '.join(missing)}")

        sizes = ", ".join(f"{p} ({Path(p).stat().st_size / 1024:,.0f} KB)" for p in outputs)
        print(f"\n  done in {elapsed:.1f}s -> {sizes}\n")

    print(f"Pipeline complete in {total:.1f}s.")
    print("Next: python src/check_data.py   (writes docs/data_check.md)")


if __name__ == "__main__":
    main()
