"""Pack the star schema into one .xlsx for upload to the Power BI service.

Power BI Desktop is Windows-only. On a Mac the practical route is to upload a
single workbook to the service and model it in the browser, so this collapses
the seven CSVs into one file with seven named Excel tables -- named tables
(not bare sheets) because Power BI imports them as cleanly-named model tables
rather than guessing at ranges.

All data is SYNTHETIC (see CLAUDE.md). Run from the project root:
    python src/export_powerbi_workbook.py
"""

from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

PROCESSED = Path("data/processed")
OUTPUT = PROCESSED / "utility_cx_model.xlsx"

TABLES = ["dim_date", "dim_team", "dim_agent", "dim_topic", "dim_channel",
          "fact_interactions", "fact_sqm_monthly"]

# Excel's hard ceiling per cell. Coaching notes and transcripts are far under
# it, but truncating loudly beats a corrupt file.
CELL_LIMIT = 32_000


def main():
    missing = [t for t in TABLES if not (PROCESSED / f"{t}.csv").exists()]
    if missing:
        raise SystemExit(f"Missing {missing} -- run: python src/build_model_tables.py")

    frames = {}
    for name in TABLES:
        df = pd.read_csv(PROCESSED / f"{name}.csv")
        for col in df.columns[df.dtypes.astype(str).isin(["object", "string", "str"])]:
            too_long = df[col].astype(str).str.len() > CELL_LIMIT
            if too_long.any():
                print(f"  truncating {int(too_long.sum())} oversized value(s) in {name}.{col}")
                df.loc[too_long, col] = df.loc[too_long, col].astype(str).str[:CELL_LIMIT]
        frames[name] = df

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        for name, df in frames.items():
            df.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            # Wrap each sheet in a named Excel table so Power BI imports it
            # under that exact name.
            ref = f"A1:{get_column_letter(df.shape[1])}{len(df) + 1}"
            table = Table(displayName=name, ref=ref)
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleLight1", showRowStripes=False)
            sheet.add_table(table)

    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\nWrote {OUTPUT} ({size_mb:.1f} MB)")
    for name, df in frames.items():
        print(f"  {name:20s} {len(df):>6,} rows x {df.shape[1]:>2} cols")

    if size_mb > 100:
        print("\n⚠ Over the 100MB service upload limit -- drop coaching_note "
              "and survey_comment, or load from OneDrive instead.")
    print(f"\nNext: upload to a Power BI workspace. See powerbi/BUILD_GUIDE.md")


if __name__ == "__main__":
    main()
