#!/usr/bin/env python3
"""
Build historic reference convergence tables.

This reconstructs the earliest translation artifact for the original
``Digital Humanities`` term from ``historic_materials/translated_dh_terms.csv``.
For that term, the ``term`` column contains the pre-existing/Wikipedia-derived
term when one was available, while ``translated_term`` contains the historic
Google Translate output. It intentionally filters to a single ``term_source`` so
the output is not mixed with later multi-term files.

Outputs are written to:
    datasets/translated_terms/digital_humanities/evaluation/

Usage:
    python scripts/exploration/build_historic_reference_convergence.py
    python scripts/exploration/build_historic_reference_convergence.py --term "Digital Humanities"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils import analyze_differences, get_data_directory_path

console = Console()

SERVICE_COLUMNS = {
    'Wikipedia-derived historic term': 'term',
    'Historic Google Translate output': 'translated_term',
}


def term_slug(term: str) -> str:
    return term.lower().replace(' ', '_')


def has_text(value) -> bool:
    value = '' if value is None else str(value).strip()
    return bool(value) and value.lower() not in {'nan', 'none', 'null'}


def normalize_translation(value) -> str:
    """Light normalization for exact agreement checks.

    This mirrors the conservative downstream posture: normalize Unicode,
    casing, spacing, and edge punctuation, but do not attempt semantic or fuzzy
    equivalence.
    """
    if not has_text(value):
        return ''
    text = unicodedata.normalize('NFKC', str(value)).casefold()
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip(' "\'“”‘’.,;:!¡?¿()[]{}')


def _filter_term(df: pd.DataFrame, term: str, path: Path) -> pd.DataFrame:
    if 'term_source' not in df.columns:
        raise ValueError(f'Missing term_source column in {path}')
    filtered = df[df['term_source'].astype(str).str.strip() == term].copy()
    if filtered.empty:
        available = sorted(df['term_source'].dropna().astype(str).unique().tolist())
        raise ValueError(
            f'No rows for term_source={term!r} in {path}. '
            f'Available term_source values: {available}'
        )
    return filtered


def _load_directionality(data_dir: Path) -> dict[str, str]:
    meta_path = data_dir / 'metadata_files' / 'language_codes_comprehensive.csv'
    if not meta_path.exists():
        return {}
    meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    if not {'language_code', 'directionality'}.issubset(meta.columns):
        return {}
    return dict(zip(meta['language_code'], meta['directionality']))


def build_historic_reference_convergence(
    data_dir: str | Path,
    term: str = 'Digital Humanities',
    source_slug: str = 'digital_humanities',
    output_dir: Optional[str | Path] = None,
) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    project_root = data_dir.parent
    source_dir = data_dir / 'translated_terms' / source_slug
    eval_dir = Path(output_dir) if output_dir else source_dir / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)

    historic_flat_path = project_root / 'historic_materials' / 'translated_dh_terms.csv'

    if not historic_flat_path.exists():
        raise FileNotFoundError(historic_flat_path)

    historic_flat_all = pd.read_csv(historic_flat_path, dtype=str, keep_default_na=False)

    historic_flat = _filter_term(historic_flat_all, term, historic_flat_path)
    directionality = _load_directionality(data_dir)

    historic_scope = pd.DataFrame([
        {
            'artifact': 'historic_materials/translated_dh_terms.csv',
            'term_source': term,
            'rows': len(historic_flat),
            'languages': historic_flat['language'].nunique(),
            'terms': historic_flat['term_source'].nunique(),
            'nonblank_translations': int(historic_flat['translated_term'].map(has_text).sum()),
            'role_in_this_section': 'early flattened historic translation output; compares existing DH terms with historic Google Translate outputs',
        },
    ])

    service_presence = []
    for service, col in SERVICE_COLUMNS.items():
        present = historic_flat[col].map(has_text) if col in historic_flat.columns else pd.Series(False, index=historic_flat.index)
        service_presence.append({
            'source': service,
            'translation_column': col,
            'languages_with_translation': int(present.sum()),
            'total_languages': historic_flat['language'].nunique(),
            'coverage_rate': present.sum() / historic_flat['language'].nunique(),
        })
    service_presence = pd.DataFrame(service_presence)

    convergence_records = []
    for _, row in historic_flat.iterrows():
        language_code = row.get('language', '')
        row_directionality = directionality.get(language_code, 'ltr') or 'ltr'

        all_values = []
        service_values = {}
        for service, col in SERVICE_COLUMNS.items():
            raw = row.get(col, '')
            norm = normalize_translation(raw)
            service_values[service] = raw if has_text(raw) else None
            if norm:
                all_values.append((service, raw, norm))

        all_unique = len({norm for service, raw, norm in all_values})

        if not all_values:
            convergence_label = 'no outputs'
        elif len(all_values) == 1:
            convergence_label = 'one output only'
        elif all_unique == 1:
            convergence_label = 'all available outputs agree'
        else:
            convergence_label = 'outputs diverge'

        differences = analyze_differences(service_values, row_directionality)

        convergence_records.append({
            'language_code': language_code,
            'language_name': row.get('language_name', ''),
            'term_source': term,
            'directionality': row_directionality,
            'wikipedia_historic_term': row.get('term', ''),
            'google_translate_historic_output': row.get('translated_term', ''),
            'all_available_sources': '; '.join(service for service, raw, norm in all_values),
            'n_sources_present': len(all_values),
            'n_distinct_translations': all_unique,
            'sources_agree': len(all_values) > 1 and all_unique == 1,
            'all_source_convergence': convergence_label,
            'difference_summary': differences.get('summary', 'unknown'),
            'difference_details': str(differences),
        })
    convergence_detail = pd.DataFrame(convergence_records)

    overlap_mask = historic_flat['term'].map(has_text) & historic_flat['translated_term'].map(has_text)
    overlap = historic_flat[overlap_mask].copy()
    if overlap.empty:
        agreement_mask = pd.Series(dtype=bool)
    else:
        agreement_mask = overlap.apply(
            lambda r: normalize_translation(r['term']) == normalize_translation(r['translated_term']),
            axis=1,
        )
    pairwise_agreement = pd.DataFrame([
        {
            'comparison': 'Wikipedia-derived historic term vs Historic Google Translate output',
            'term_source': term,
            'overlap': len(overlap),
            'exact_normalized_agreements': int(agreement_mask.sum()),
            'agreement_rate': agreement_mask.mean() if len(overlap) else 0,
        },
    ])

    ref_count_summary = (
        convergence_detail
        .groupby('n_sources_present', as_index=False)
        .agg(
            languages=('language_code', 'nunique'),
            sources_agree=('sources_agree', 'sum'),
            outputs_all_agree=('sources_agree', 'sum'),
        )
    )
    ref_count_summary['term_source'] = term
    ref_count_summary['agreement_rate'] = (
        ref_count_summary['sources_agree'] / ref_count_summary['languages']
    )

    all_source_summary = (
        convergence_detail['all_source_convergence']
        .value_counts()
        .rename_axis('convergence_category')
        .reset_index(name='languages')
    )
    all_source_summary['term_source'] = term
    all_source_summary['share'] = all_source_summary['languages'] / len(convergence_detail)

    outputs = {
        'historic_reference_artifacts': historic_scope,
        'historic_reference_service_presence': service_presence,
        'historic_pairwise_reference_agreement': pairwise_agreement,
        'historic_reference_count_summary': ref_count_summary,
        'historic_all_source_convergence_summary': all_source_summary,
        'historic_reference_convergence_detail': convergence_detail,
    }

    for name, df in outputs.items():
        df.to_csv(eval_dir / f'{name}.csv', index=False)

    return outputs


def print_summary(outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    table = Table(title='Historic Reference Convergence')
    table.add_column('Comparison')
    table.add_column('Overlap', justify='right')
    table.add_column('Agreements', justify='right')
    table.add_column('Agreement rate', justify='right')

    pairwise = outputs['historic_pairwise_reference_agreement']
    for _, row in pairwise.iterrows():
        table.add_row(
            row['comparison'],
            f"{int(row['overlap']):,}",
            f"{int(row['exact_normalized_agreements']):,}",
            f"{row['agreement_rate']:.1%}",
        )
    console.print(table)
    console.print(f'[green]Wrote outputs to:[/green] {output_dir}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--term', default='Digital Humanities')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--source-slug', default='digital_humanities')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir or get_data_directory_path())
    source_dir = data_dir / 'translated_terms' / args.source_slug
    output_dir = Path(args.output_dir) if args.output_dir else source_dir / 'evaluation'

    outputs = build_historic_reference_convergence(
        data_dir=data_dir,
        term=args.term,
        source_slug=args.source_slug,
        output_dir=output_dir,
    )
    print_summary(outputs, output_dir)


if __name__ == '__main__':
    main()
