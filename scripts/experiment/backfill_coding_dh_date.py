#!/usr/bin/env python3
"""
One-off script: add coding_dh_date to translation CSVs that are missing it.

Uses each file's modification time as the timestamp so the value is as accurate
as possible. Skips evaluation output files (old_digital_humanities/evaluation/).
"""

import os
import sys
import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils import get_data_directory_path

SKIP_DIRS = {'evaluation'}

def backfill(base_dir: str, dry_run: bool = False):
    print(base_dir)
    fixed, skipped = 0, 0
    for root, dirs, files in os.walk(base_dir):
        # Skip evaluation subdirectories
        print(dirs)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if not fname.endswith('.csv'):
                continue
            path = os.path.join(root, fname)
            df = pd.read_csv(path)
            if 'coding_dh_date' in df.columns and df['coding_dh_date'].notna().all():
                skipped += 1
                continue
            mtime = os.path.getmtime(path)
            timestamp = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            if 'coding_dh_date' not in df.columns:
                df.insert(0, 'coding_dh_date', timestamp)
            else:
                df['coding_dh_date'] = df['coding_dh_date'].fillna(timestamp)
            rel = os.path.relpath(path, base_dir)
            if dry_run:
                print(f"[dry run] would add coding_dh_date={timestamp}  {rel}")
            else:
                df.to_csv(path, index=False)
                print(f"fixed  {rel}  ({timestamp})")
            fixed += 1
    print(f"\n{'Would fix' if dry_run else 'Fixed'} {fixed} files, skipped {skipped} already-dated files.")

if __name__ == '__main__':
    base = os.path.join(get_data_directory_path(), 'translated_terms/digital_humanities')
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        print("--- DRY RUN ---\n")
    backfill(base, dry_run=dry_run)
