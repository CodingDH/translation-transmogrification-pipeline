#!/usr/bin/env python3
"""
One-off script: add family_name to translation CSVs that are missing it.

Loads the comprehensive language codes CSV (which includes the JSON family
fallback applied automatically by load_language_codes()), then merges
family_name into every translation CSV under translated_terms/digital_humanities/.

Delete this file once all CSVs have been updated.

Usage:
    python backfill_language_family.py           # live run
    python backfill_language_family.py --dry-run # preview only
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils import get_data_directory_path, read_csv_file
from scripts.experiment.generate_language_codes import load_language_codes

SKIP_DIRS = {'evaluation'}


def backfill(base_dir: str, dry_run: bool = False) -> None:
    lang_df = load_language_codes()
    # Build code → family_name lookup (use English language name column as renamed by load_language_codes)
    family_map = (
        lang_df[['language_code', 'family_name']]
        .dropna(subset=['language_code'])
        .drop_duplicates('language_code')
        .set_index('language_code')['family_name']
        .fillna('Other')
    )

    fixed, skipped = 0, 0
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in sorted(files):
            if not fname.endswith('.csv'):
                continue
            path = os.path.join(root, fname)
            df = read_csv_file(path, error_bad_lines=False)

            if 'language_code' not in df.columns:
                print(f"skip   {os.path.relpath(path, base_dir)}  (no language_code column)")
                skipped += 1
                continue

            if 'family_name' in df.columns and df['family_name'].notna().all() \
                    and (df['family_name'] != 'Other').all():
                skipped += 1
                continue

            df['family_name'] = df['language_code'].astype(str).map(family_map).fillna('Other')

            rel = os.path.relpath(path, base_dir)
            if dry_run:
                sample = df[['language_code', 'family_name']].drop_duplicates().head(5).to_string(index=False)
                print(f"[dry run] {rel}\n{sample}\n")
            else:
                df.to_csv(path, index=False)
                print(f"fixed  {rel}")
            fixed += 1

    print(f"\n{'Would fix' if dry_run else 'Fixed'} {fixed} files, skipped {skipped}.")


if __name__ == '__main__':
    base = os.path.join(get_data_directory_path(), 'translated_terms/digital_humanities')
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        print("--- DRY RUN ---\n")
    backfill(base, dry_run=dry_run)
