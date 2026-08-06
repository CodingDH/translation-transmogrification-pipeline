#!/usr/bin/env python3
"""
Build a reviewable mapping from reconciled family labels to analysis labels.

The language metadata intentionally keeps multiple family layers:

* ``family_name`` is the raw ISO/CLDR/manual/JSON build trace.
* ``family_name_reconciled`` is the more specific reviewed/Glottolog-aware label.
* ``family_name_analysis`` defaults to ``family_name_reconciled`` unless a
  reviewer explicitly accepts a normalization or override.

This script creates the review surface for the third layer. It preserves
specific Glottolog/reconciled family names by default and flags only labels that
need a label-level decision here: blank labels, duplicated spellings, and
special non-family categories. Problematic macrofamily/geographic labels are
marked as row-level reconciliation signals, not as mapping decisions. It does not modify
``language_codes_comprehensive.csv``.

Usage:
    python scripts/exploration/build_family_analysis_mapping.py
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


DEFAULT_OUTPUT = 'family_analysis_mapping.csv'


NORMALIZE_TO_EXISTING = {
    'Afro-Asiatic': 'Afro-Asiatic languages',
    'Austroasiatic': 'Austro-Asiatic languages',
    'Austronesian': 'Austronesian languages',
    'Dravidian': 'Dravidian languages',
    'Eskimo-Aleut': 'Eskimo-Aleut languages',
    'Indo-European': 'Indo-European languages',
    'Tai-Kadai': 'Tai-Kadai languages',
}


ROW_LEVEL_RECONCILIATION_LABELS = {
    'Altaic languages': 'Contested macrofamily; assign affected languages to more specific families in the reconciliation review.',
    'Central American Indian languages': 'ISO 639-5 geographic cover label; assign affected languages to genealogical families in the reconciliation review.',
    'Niger-Kordofanian languages': 'Legacy macrofamily label; assign affected languages to more specific Glottolog/SIL-informed families in the reconciliation review.',
    'Nilo-Saharan languages': 'Contested macrofamily; assign affected languages to more specific families in the reconciliation review.',
    'North American Indian languages': 'ISO 639-5 geographic cover label; assign affected languages to genealogical families in the reconciliation review.',
    'South American Indian languages': 'ISO 639-5 geographic cover label; assign affected languages to genealogical families in the reconciliation review.',
}


SPECIAL_EXCLUDE = {
    'Undeciphered script',
}


SPECIAL_RETAIN = {
    'Artificial languages',
    'Creoles and pidgins',
    'Language isolate',
    'Sign languages',
}


def propose_analysis_label(family: str) -> tuple[str, str, str, str]:
    """Return ``(analysis_label, reason, action, review_status)`` for a family."""
    family = str(family or '').strip()
    if not family:
        return 'Unassigned', 'Blank family label; keep visible for QA.', 'missing_family', 'pending'
    if family in NORMALIZE_TO_EXISTING:
        target = NORMALIZE_TO_EXISTING[family]
        return target, f'Normalizes variant spelling to existing label: {target}.', 'normalize_variant', 'pending'
    if family in ROW_LEVEL_RECONCILIATION_LABELS:
        return family, ROW_LEVEL_RECONCILIATION_LABELS[family], 'row_level_reconciliation', 'reference_only'
    if family in SPECIAL_EXCLUDE:
        return 'Exclude from family aggregation', 'Special non-family category; keep out of family-comparison charts.', 'exclude_special', 'pending'
    if family in SPECIAL_RETAIN:
        return family, 'Special comparison category retained as its own visible analysis label.', 'retain_special_category', 'accepted'
    return family, 'Preserves reconciled family label for analysis.', 'preserve_reconciled', 'accepted'


def sample_languages(rows: pd.DataFrame, limit: int = 8) -> str:
    examples = []
    for _, row in rows.sort_values(['language_name', 'language_code']).head(limit).iterrows():
        examples.append(f"{row['language_code']} ({row['language_name']})")
    return '; '.join(examples)


def build_mapping(metadata_dir: Path, output_path: Path | None = None) -> pd.DataFrame:
    comp_path = metadata_dir / 'language_codes_comprehensive.csv'
    if not comp_path.exists():
        raise FileNotFoundError(f'Missing {comp_path}')

    df = pd.read_csv(comp_path, converters={'language_code': str}, keep_default_na=False)
    rows = []
    for family, group in df.groupby('family_name_reconciled', dropna=False):
        analysis, reason, action, review_status = propose_analysis_label(family)
        raw_families = (
            group['family_name']
            .fillna('')
            .replace('', '(blank)')
            .value_counts()
            .rename_axis('family_name')
            .reset_index(name='count')
        )
        raw_summary = '; '.join(
            f"{r.family_name}: {int(r['count'])}" for _, r in raw_families.iterrows()
        )
        rows.append({
            'family_name_reconciled': family,
            'family_name_analysis': analysis,
            'n_languages': len(group),
            'sample_languages': sample_languages(group),
            'raw_family_names': raw_summary,
            'mapping_action': action,
            'mapping_reason': reason,
            'review_status': review_status,
            'reviewer_notes': '',
        })

    out = pd.DataFrame(rows).sort_values(
        ['family_name_analysis', 'n_languages', 'family_name_reconciled'],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    output_path = output_path or metadata_dir / DEFAULT_OUTPUT
    out.to_csv(output_path, index=False)
    return out


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    summary = (
        df.groupby('family_name_analysis')
        .agg(
            reconciled_families=('family_name_reconciled', 'nunique'),
            languages=('n_languages', 'sum'),
        )
        .sort_values('languages', ascending=False)
        .reset_index()
    )

    table = Table(title='Family Analysis Mapping Proposal')
    table.add_column('Analysis bucket')
    table.add_column('Families', justify='right')
    table.add_column('Languages', justify='right')
    for _, row in summary.iterrows():
        table.add_row(
            str(row['family_name_analysis']),
            str(int(row['reconciled_families'])),
            str(int(row['languages'])),
        )
    console.print(table)
    action_counts = df['mapping_action'].value_counts().rename_axis('mapping_action').reset_index(name='rows')
    action_table = Table(title='Mapping Actions')
    action_table.add_column('Action')
    action_table.add_column('Rows', justify='right')
    for _, row in action_counts.iterrows():
        action_table.add_row(str(row['mapping_action']), str(int(row['rows'])))
    console.print(action_table)
    console.print(f'Wrote: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build family analysis mapping review CSV.')
    parser.add_argument('--metadata-dir', type=Path, default=None)
    parser.add_argument('--output-path', type=Path, default=None)
    args = parser.parse_args()

    metadata_dir = args.metadata_dir or Path(get_data_directory_path()) / 'metadata_files'
    output_path = args.output_path or metadata_dir / DEFAULT_OUTPUT
    df = build_mapping(metadata_dir, output_path)
    print_summary(df, output_path)


if __name__ == '__main__':
    main()
