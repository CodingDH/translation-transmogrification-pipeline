"""Audit mixed language identifiers in language_codes_comprehensive.csv.

This is a lightweight diagnostic script. It does not rerun translations or alter
existing results. It reports where project-level language_code differs from the
preferred BCP 47-style tag added by add_identifier_context() in
generate_language_codes.py, and counts per-service support coverage from any
{service}_supported columns added by add_service_language_codes().
"""

from __future__ import annotations

import argparse
import re
import pandas as pd

MIXED_CODE_RE = re.compile(r"[-_]")


def main(path: str) -> None:
    # na_filter=False keeps the literal string 'nan' (Min Nan Chinese / ISO 639-3)
    # from being silently converted into NaN by pandas.
    df = pd.read_csv(path, converters={"language_code": str}, na_filter=False)
    total = len(df)
    mixed = df[df["language_code"].str.contains(MIXED_CODE_RE, na=False)]
    if "bcp47_tag" in df.columns:
        changed = df[df["bcp47_tag"] != df["language_code"].str.replace("_", "-", regex=False)]
    else:
        changed = pd.DataFrame()

    print(f"Rows: {total}")
    print(f"Hyphen/underscore project codes: {len(mixed)}")
    if len(mixed):
        cols = [c for c in ["language_code", "language_name", "bcp47_tag", "bcp47_source", "bcp47_note"] if c in df.columns]
        print("\nHyphen/underscore project codes:")
        print(mixed[cols].to_string(index=False))
    if "bcp47_tag" in df.columns:
        print(f"\nRows where bcp47_tag differs from language_code (after underscore normalization): {len(changed)}")
        cols = [c for c in ["language_code", "language_name", "bcp47_tag", "bcp47_source", "bcp47_note"] if c in df.columns]
        print(changed[cols].head(50).to_string(index=False))
    else:
        print("\nNo bcp47_tag column found. Regenerate or load with updated script to add identifier context.")

    for col in [c for c in df.columns if c.endswith("_supported")]:
        print(f"{col}: {(df[col].astype(str).str.lower() == 'true').sum()}/{total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to language_codes_comprehensive.csv")
    args = parser.parse_args()
    main(args.csv_path)
