"""
Audit mixed language identifiers in language_codes_comprehensive.csv.

This is a lightweight diagnostic script. It does not rerun translations or alter existing results. It reports where project-level language_code differs from the preferred BCP 47-style tag added by add_identifier_context() in generate_language_codes.py, and counts per-service support coverage from any {service}_supported columns added by add_service_language_codes().

By default, reads from <data_dir>/metadata_files language_codes_comprehensive.csv where <data_dir> comes from the apikey-stored CODING_DH_DATA_DIRECTORY_PATH. A positional `csv_path` argument can override that default for ad-hoc audits.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd
from rich.console import Console

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from scripts.utils import get_data_directory_path, read_csv_file

console = Console()

MIXED_CODE_RE = re.compile(r"[-_]")
BCP47_COLS = ["language_code", "language_name", "bcp47_tag", "bcp47_source", "bcp47_note"]


def default_csv_path() -> str:
    """Return the canonical path to ``language_codes_comprehensive.csv``.

    Resolves via ``scripts.utils.get_data_directory_path()`` so the script points at the same data directory the rest of the pipeline uses, regardless of which working directory it was invoked from.
    """
    return os.path.join(get_data_directory_path(), 'metadata_files', 'language_codes_comprehensive.csv')


def load_comprehensive_csv(path: str) -> pd.DataFrame:
    """Load ``language_codes_comprehensive.csv`` via the project-wide reader.

    Delegates to ``scripts.utils.read_csv_file``, which already applies
    ``converters={'language_code': str}`` to preserve the literal string
    ``'nan'`` (the ISO 639-3 code for Min Nan Chinese) and walks the
    UTF-8 → latin-1 → Python-engine fallback chain so audits on legacy or
    manually-edited CSVs don't fail on a single mis-encoded row.

    Empty string fields (e.g. ``bcp47_note`` for rows with no annotation) are coerced back from NaN so the printed audit tables show blank cells rather than the noisy ``NaN`` literal.
    """
    return read_csv_file(path).fillna("")


def report_hyphen_underscore_codes(df: pd.DataFrame) -> None:
    """Print the rows whose ``language_code`` contains a hyphen or underscore.

    These are the rows most likely to carry interesting BCP 47 rewrites or grandfathered fallbacks (e.g. ``bat-smg`` → ``sgs``, ``uz_AF`` → ``uz-AF``). Showing them up-front gives a quick sanity check that the BCP 47 layer is handling Wikimedia legacy codes the way it should be.
    """
    mixed = df[df["language_code"].str.contains(MIXED_CODE_RE, na=False)]
    console.print(f"Hyphen/underscore project codes: {len(mixed)}")
    if len(mixed):
        cols = [c for c in BCP47_COLS if c in df.columns]
        console.print("\nHyphen/underscore project codes:")
        console.print(mixed[cols].to_string(index=False))


def report_bcp47_changes(df: pd.DataFrame, head: int = 50) -> None:
    """Print rows where ``bcp47_tag`` differs from the hyphen-normalised project code.

    Comparison is against ``language_code.str.replace('_', '-')`` so locale variants that differ only in separator (``uz_AF`` → ``uz-AF``) don't get flagged. What's left are substantive rewrites: IANA Preferred-Value redirects (``iw`` → ``he``, ``mo`` → ``ro``) and the grandfathered Wikimedia fallbacks (``bat-smg`` → ``sgs``).

    ``head`` caps the number of rows printed so very large diffs stay readable.
    """
    if "bcp47_tag" not in df.columns:
        console.print(
            "\nNo bcp47_tag column found. Regenerate or load with updated "
            "script to add identifier context."
        )
        return
    changed = df[df["bcp47_tag"] != df["language_code"].str.replace("_", "-", regex=False)]
    console.print(
        f"\nRows where bcp47_tag differs from language_code "
        f"(after underscore normalization): {len(changed)}"
    )
    cols = [c for c in BCP47_COLS if c in df.columns]
    console.print(changed[cols].head(head).to_string(index=False))


def report_service_support(df: pd.DataFrame) -> None:
    """Print per-service support coverage for every ``*_supported`` column.

    Each column named ``<service>_supported`` (added by ``add_service_language_codes()``) is summed by counting ``'true'``-ish values, printing a ``<service>: matched/total`` line. Useful for confirming that a refresh of ``service_language_code_support.csv`` produced the expected coverage shift.
    """
    total = len(df)
    for col in [c for c in df.columns if c.endswith("_supported")]:
        n_true = (df[col].astype(str).str.lower() == 'true').sum()
        console.print(f"{col}: {n_true}/{total}")


def main(path: str | None = None) -> None:
    """Run the full audit against ``path`` (or the project default if ``None``).

    Wraps the three reporters above with the standard ``Source:`` and ``Rows:``header and the canonical CSV-loading helper. Pass an explicit ``path`` when auditing a backup or non-canonical CSV; pass ``None`` (or omit) for the project's standard ``datasets/metadata_files/language_codes_comprehensive.csv``.
    """
    if path is None:
        path = default_csv_path()
    df = load_comprehensive_csv(path)
    console.print(f"Source: {path}")
    console.print(f"Rows: {len(df)}")
    report_hyphen_underscore_codes(df)
    report_bcp47_changes(df)
    report_service_support(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help=(
            "Path to language_codes_comprehensive.csv. "
            "If omitted, defaults to the project data directory's copy."
        ),
    )
    args = parser.parse_args()
    main(args.csv_path)
