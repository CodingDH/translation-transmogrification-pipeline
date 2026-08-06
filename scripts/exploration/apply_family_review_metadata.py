#!/usr/bin/env python3
"""
Apply reviewed family metadata without rebuilding the language-code source list.

This is a post-build analytical metadata layer. It leaves
``language_codes_comprehensive.csv`` untouched and writes a reviewed copy that
can be used by notebooks and downstream analysis after family review.

Inputs:
    datasets/metadata_files/language_codes_comprehensive.csv
    datasets/metadata_files/family_analysis_mapping_reviewed.csv
    datasets/metadata_files/family_reconciliation_reviewed.csv  (optional)

Output:
    datasets/metadata_files/language_codes_comprehensive_family_reviewed.csv

Usage:
    python scripts/exploration/apply_family_review_metadata.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils import get_data_directory_path

console = Console()


DEFAULT_OUTPUT = 'language_codes_comprehensive_family_reviewed.csv'


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _norm(value: object) -> str:
    return str(value or '').strip()


def _iso639_3(row: pd.Series) -> str:
    for col in ('language_code', 'iso639_2_t'):
        value = _norm(row.get(col, ''))
        if len(value) == 3:
            return value
    return ''


def _load_glottolog_family_map(metadata_dir: Path) -> dict[str, str]:
    glottolog_path = metadata_dir / 'glottolog-cache' / 'languoid.csv'
    if not glottolog_path.exists():
        return {}
    glot = pd.read_csv(glottolog_path, low_memory=False)
    id_to_name = glot.set_index('id')['name'].to_dict()
    glot['glottolog_family'] = glot['family_id'].map(id_to_name)
    glot_lang = glot[(glot['level'] == 'language') & glot['iso639P3code'].notna()].copy()
    return glot_lang.set_index('iso639P3code')['glottolog_family'].fillna('').astype(str).to_dict()


def _load_mapping(metadata_dir: Path) -> tuple[dict[str, str], str]:
    reviewed = metadata_dir / 'family_analysis_mapping_reviewed.csv'
    generated = metadata_dir / 'family_analysis_mapping.csv'
    mapping_path = reviewed if reviewed.exists() else generated
    if not mapping_path.exists():
        return {}, ''

    mapping_df = _read_csv(mapping_path)
    required = {'family_name_reconciled', 'family_name_analysis'}
    if not required.issubset(mapping_df.columns):
        raise ValueError(f'{mapping_path} must contain {sorted(required)}')

    mapping = {}
    for _, row in mapping_df.iterrows():
        source = _norm(row.get('family_name_reconciled', ''))
        target = _norm(row.get('family_name_analysis', ''))
        if source and target:
            mapping[source] = target
    return mapping, mapping_path.name


def _apply_reconciliation_review(df: pd.DataFrame, metadata_dir: Path) -> tuple[pd.DataFrame, int, int, str]:
    review_path = metadata_dir / 'family_reconciliation_reviewed.csv'
    if not review_path.exists():
        return df, 0, 0, ''

    review = _read_csv(review_path)
    if 'review_kind' not in review.columns:
        return df, 0, 0, review_path.name

    df = df.copy()
    pair_applied = 0
    language_applied = 0

    pair_rows = review[
        review['review_kind'].eq('iso_glottolog_pair')
        & review['reviewer_family_name'].astype(str).str.strip().ne('')
    ].copy()
    if not pair_rows.empty:
        glottolog_family_by_iso3 = _load_glottolog_family_map(metadata_dir)
        df['_family_review_iso3'] = df.apply(_iso639_3, axis=1)
        df['_family_review_glottolog_family'] = df['_family_review_iso3'].map(glottolog_family_by_iso3).fillna('')
        pair_lookup = {
            (_norm(row['family_name_iso']), _norm(row['family_name_glottolog'])): _norm(row['reviewer_family_name'])
            for _, row in pair_rows.iterrows()
        }
        for idx, row in df.iterrows():
            key = (_norm(row.get('family_name', '')), _norm(row.get('_family_review_glottolog_family', '')))
            reviewed_family = pair_lookup.get(key)
            if reviewed_family:
                df.at[idx, 'family_name_reconciled_reviewed'] = reviewed_family
                df.at[idx, 'family_review_source'] = 'family_reconciliation_reviewed:iso_glottolog_pair'
                pair_applied += 1
        df = df.drop(columns=['_family_review_iso3', '_family_review_glottolog_family'], errors='ignore')

    language_rows = review[
        review['review_kind'].eq('json_fallback_language')
        & review['reviewer_decision'].astype(str).str.lower().eq('revise')
        & review['reviewer_family_name'].astype(str).str.strip().ne('')
    ].copy()
    if not language_rows.empty:
        language_lookup = {
            _norm(row['language_code']): {
                'family': _norm(row['reviewer_family_name']),
                'notes': _norm(row.get('reviewer_notes', '')),
            }
            for _, row in language_rows.iterrows()
        }
        for idx, row in df.iterrows():
            code = _norm(row.get('language_code', ''))
            reviewed = language_lookup.get(code)
            if reviewed:
                df.at[idx, 'family_name_reconciled_reviewed'] = reviewed['family']
                df.at[idx, 'family_review_source'] = 'family_reconciliation_reviewed:json_fallback_language'
                df.at[idx, 'family_review_notes'] = reviewed['notes']
                language_applied += 1

    return df, pair_applied, language_applied, review_path.name


def apply_family_review_metadata(
    metadata_dir: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    metadata_dir = Path(metadata_dir)
    input_path = metadata_dir / 'language_codes_comprehensive.csv'
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = _read_csv(input_path)
    if 'family_name_reconciled' not in df.columns:
        raise ValueError(f'{input_path} must contain family_name_reconciled')

    df['family_name_reconciled_pre_review'] = df['family_name_reconciled']
    df['family_name_reconciled_reviewed'] = df['family_name_reconciled'].where(
        df['family_name_reconciled'].astype(str).str.strip().ne(''),
        df.get('family_name', ''),
    )
    df['family_review_source'] = 'pre_review_reconciled'
    df['family_review_notes'] = ''

    df, pair_applied, language_applied, review_source = _apply_reconciliation_review(df, metadata_dir)

    mapping, mapping_source = _load_mapping(metadata_dir)
    df['family_name_analysis'] = df['family_name_reconciled_reviewed'].map(mapping).fillna(
        df['family_name_reconciled_reviewed']
    )
    df['family_name_analysis_source'] = mapping_source or 'family_name_reconciled_reviewed'

    output_path = Path(output_path) if output_path else metadata_dir / DEFAULT_OUTPUT
    df.to_csv(output_path, index=False)
    df.attrs['family_review_pair_applied'] = pair_applied
    df.attrs['family_review_language_applied'] = language_applied
    df.attrs['family_reconciliation_review_source'] = review_source
    df.attrs['family_analysis_mapping_source'] = mapping_source
    df.attrs['output_path'] = str(output_path)
    return df


def print_summary(df: pd.DataFrame) -> None:
    table = Table(title='Reviewed Family Metadata')
    table.add_column('Metric')
    table.add_column('Value', justify='right')
    table.add_row('Rows', f'{len(df):,}')
    table.add_row('Pair-level overrides applied', f"{df.attrs.get('family_review_pair_applied', 0):,}")
    table.add_row('Language-level overrides applied', f"{df.attrs.get('family_review_language_applied', 0):,}")
    changed = (
        df['family_name_reconciled_reviewed'].astype(str)
        != df['family_name_reconciled_pre_review'].astype(str)
    ).sum()
    table.add_row('Rows with reviewed reconciled-family change', f'{int(changed):,}')
    table.add_row('Analysis labels', f"{df['family_name_analysis'].nunique():,}")
    table.add_row('Reconciliation review source', df.attrs.get('family_reconciliation_review_source') or '(none)')
    table.add_row('Analysis mapping source', df.attrs.get('family_analysis_mapping_source') or '(none)')
    console.print(table)
    console.print(f"[green]Wrote:[/green] {df.attrs.get('output_path')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', type=Path, default=None)
    parser.add_argument('--output-path', type=Path, default=None)
    args = parser.parse_args()

    data_dir = Path(get_data_directory_path())
    metadata_dir = args.metadata_dir or data_dir / 'metadata_files'
    df = apply_family_review_metadata(metadata_dir, args.output_path)
    print_summary(df)


if __name__ == '__main__':
    main()
