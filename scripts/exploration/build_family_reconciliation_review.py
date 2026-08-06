#!/usr/bin/env python3
"""
Build one family reconciliation review queue.

This combines two review surfaces that belong to the same methodological step:

* ISO/Glottolog disagreement pairs from ``family_reconciliation.csv``.
* Language-level JSON fallback assignments that need spot-checking because the
  raw family assignment came from ``language_family_assignments.json``.

If ``family_analysis_mapping_reviewed.csv`` exists, its label policy is used to
annotate JSON fallback rows whose current reconciled family is still marked for
review. Otherwise the generated ``family_analysis_mapping.csv`` is used.

Output:
    datasets/metadata_files/family_reconciliation_review.csv

Usage:
    python scripts/exploration/build_family_reconciliation_review.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.experiment.generate_language_codes import MANUAL_LANG_TO_SET5, _norm_family_name
from scripts.utils import get_data_directory_path

console = Console()


DEFAULT_OUTPUT = 'family_reconciliation_review.csv'


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _code_value(value) -> str:
    if pd.isna(value):
        return 'nan'
    return str(value).strip()


def _load_glottolog_family_map(glottolog_path: Path) -> dict[str, str]:
    glot = pd.read_csv(glottolog_path, low_memory=False)
    id_to_name = glot.set_index('id')['name'].to_dict()
    glot['glottolog_family'] = glot['family_id'].map(id_to_name)
    glot_lang = glot[(glot['level'] == 'language') & glot['iso639P3code'].notna()].copy()
    return glot_lang.set_index('iso639P3code')['glottolog_family'].to_dict()


def _glottolog_status(raw_family: str, reconciled_family: str, glottolog_family: str) -> str:
    if not glottolog_family:
        return 'No Glottolog match'
    if _norm_family_name(raw_family) == _norm_family_name(glottolog_family):
        return 'Glottolog agreed'
    if raw_family != reconciled_family:
        return 'Glottolog reclassified'
    return 'Glottolog disagreed; reconciliation kept ISO/JSON family'


def _review_tier(status: str, has_cldr_hierarchy: bool) -> str:
    if status == 'No Glottolog match':
        return 'REVIEW_MED'
    if status == 'Glottolog disagreed; reconciliation kept ISO/JSON family':
        return 'REVIEW_MED'
    if not has_cldr_hierarchy:
        return 'REVIEW_MED'
    return 'REFERENCE_ONLY'


def _suggest_coherent_family(row: pd.Series, json_entry: dict, status: str) -> dict[str, str]:
    """Suggest a coherence review target without changing source metadata."""
    language_code = _code_value(row.get('language_code', ''))
    language_name = str(row.get('language_name', '') or '')
    raw_family = str(row.get('family_name_reconciled', '') or row.get('family_name', '') or '')
    json_iso = str(json_entry.get('iso639_5', '') or '')
    rationale = str(json_entry.get('rationale', '') or '')
    haystack = f'{language_code} {language_name} {json_iso} {raw_family} {rationale}'.lower()

    suggestion = {
        'coherence_issue': '',
        'suggested_iso639_5': '',
        'suggested_family_name': '',
        'suggestion_confidence': '',
        'suggestion_reason': '',
    }

    def set_suggestion(issue: str, iso: str, family: str, confidence: str, reason: str) -> dict[str, str]:
        suggestion.update({
            'coherence_issue': issue,
            'suggested_iso639_5': iso,
            'suggested_family_name': family,
            'suggestion_confidence': confidence,
            'suggestion_reason': reason,
        })
        return suggestion

    if language_code == 'frm':
        return set_suggestion(
            'incorrect-json-direct-family',
            'roa',
            'Indo-European languages',
            'high',
            'Middle French is Romance; JSON stores gem even though its rationale says to reassign to roa.',
        )

    if language_code == 'lab':
        return set_suggestion(
            'special-non-family-category',
            '',
            'Undeciphered script',
            'high',
            'Linear A is an undeciphered writing system; keep as a special non-family category or exclude from family aggregation.',
        )

    if raw_family == 'Altaic languages':
        if 'mongolic' in haystack or language_code == 'bua':
            return set_suggestion(
                'contested-macrofamily',
                'xgn',
                'Mongolic-Khitan',
                'medium',
                'Altaic is rejected elsewhere in the reconciliation policy; rationale points to Mongolic.',
            )
        if 'turk' in haystack or language_code == 'ota':
            return set_suggestion(
                'contested-macrofamily',
                'trk',
                'Turkic',
                'high',
                'Altaic is rejected elsewhere in the reconciliation policy; rationale points to Turkic.',
            )
        return set_suggestion(
            'contested-macrofamily',
            '',
            '',
            'review',
            'Altaic is rejected elsewhere in the reconciliation policy, but this row needs a specific replacement.',
        )

    if raw_family == 'Nilo-Saharan languages':
        if 'nilotic' in haystack or language_code in {'din', 'kln'}:
            return set_suggestion(
                'contested-macrofamily',
                'ssa',
                'Nilotic',
                'medium',
                'Nilo-Saharan is treated as contested elsewhere; rationale points to Nilotic.',
            )
        return set_suggestion(
            'contested-macrofamily',
            '',
            '',
            'review',
            'Nilo-Saharan is treated as contested elsewhere, but this row needs a specific replacement.',
        )

    if raw_family == 'Niger-Kordofanian languages':
        if 'mande' in haystack:
            return set_suggestion(
                'abandoned-macrofamily',
                'dmn',
                'Mande',
                'medium',
                'Niger-Kordofanian is abandoned elsewhere; rationale points to Mande.',
            )
        if 'kru' in haystack:
            return set_suggestion(
                'abandoned-macrofamily',
                'kro',
                'Kru',
                'medium',
                'Niger-Kordofanian is abandoned elsewhere; rationale points to Kru.',
            )
        if any(term in haystack for term in ['bantu', 'kwa', 'akan', 'ubangian', 'niger-congo']):
            return set_suggestion(
                'abandoned-macrofamily',
                'alv',
                'Atlantic-Congo',
                'medium',
                'Niger-Kordofanian is abandoned elsewhere; rationale points toward Atlantic-Congo/Niger-Congo branches.',
            )
        return set_suggestion(
            'abandoned-macrofamily',
            '',
            '',
            'review',
            'Niger-Kordofanian is abandoned elsewhere, but this row needs a specific replacement.',
        )

    if raw_family == 'North American Indian languages':
        if 'algonquian' in haystack or language_code == 'del':
            return set_suggestion(
                'geographic-grouping',
                'alg',
                'Algic',
                'high',
                'North American Indian is geographic; rationale points to Algonquian/Algic.',
            )
        if 'athabaskan' in haystack or language_code == 'den':
            return set_suggestion(
                'geographic-grouping',
                'ath',
                'Athabaskan-Eyak-Tlingit',
                'medium',
                'North American Indian is geographic; rationale points to Athabaskan.',
            )
        return set_suggestion(
            'geographic-grouping',
            '',
            '',
            'review',
            'North American Indian is geographic, but this row needs a specific replacement.',
        )

    if raw_family == 'South American Indian languages':
        if 'tupian' in haystack or language_code == 'gub':
            return set_suggestion(
                'geographic-grouping',
                'tup',
                'Tupian',
                'medium',
                'South American Indian is geographic; rationale points to Tupian.',
            )
        return set_suggestion(
            'geographic-grouping',
            '',
            '',
            'review',
            'South American Indian is geographic, but this row needs a specific replacement.',
        )

    if raw_family == 'Central American Indian languages':
        if 'oto-manguean' in haystack or 'zapotec' in haystack or language_code == 'zap':
            return set_suggestion(
                'geographic-grouping',
                'omq',
                'Otomanguean',
                'medium',
                'Central American Indian is geographic; rationale points to Zapotec/Oto-Manguean.',
            )
        return set_suggestion(
            'geographic-grouping',
            '',
            '',
            'review',
            'Central American Indian is geographic, but this row needs a specific replacement.',
        )

    if status == 'No Glottolog match':
        return set_suggestion(
            'no-external-validation',
            '',
            raw_family,
            'low',
            'No Glottolog match; broad family is plausible but not externally cross-validated in this pipeline.',
        )

    return suggestion


def _load_analysis_policy(metadata_dir: Path) -> tuple[dict[str, dict[str, str]], str]:
    reviewed = metadata_dir / 'family_analysis_mapping_reviewed.csv'
    generated = metadata_dir / 'family_analysis_mapping.csv'
    policy_path = reviewed if reviewed.exists() else generated
    if not policy_path.exists():
        return {}, ''

    policy = _read_csv(policy_path)
    policy_by_family = {}
    for _, row in policy.iterrows():
        family = str(row.get('family_name_reconciled', '')).strip()
        if not family:
            continue
        policy_by_family[family] = {
            'analysis_family_name': str(row.get('family_name_analysis', '')).strip(),
            'analysis_mapping_action': str(row.get('mapping_action', '')).strip(),
            'analysis_review_status': str(row.get('review_status', '')).strip(),
            'analysis_mapping_reason': str(row.get('mapping_reason', '')).strip(),
        }
    return policy_by_family, policy_path.name


def _base_review_columns(row: dict[str, object]) -> dict[str, object]:
    base = {
        'review_kind': '',
        'case_id': '',
        'review_tier': '',
        'reviewer_decision': '',
        'reviewer_family_name': '',
        'reviewer_iso639_5': '',
        'reviewer_notes': '',
        'language_code': '',
        'language_name': '',
        'sources': '',
        'iso639_2_t': '',
        'n_languages': '',
        'sample_languages': '',
        'family_name_iso': '',
        'family_name_glottolog': '',
        'choice': '',
        'category': '',
        'chosen_family_name': '',
        'rationale': '',
        'json_iso639_5': '',
        'json_family_name': '',
        'raw_family_name': '',
        'family_name_reconciled': '',
        'glottolog_family': '',
        'glottolog_validation_status': '',
        'glottolog_changed_family': '',
        'has_cldr_iso6395_hierarchy': '',
        'coherence_issue': '',
        'suggested_iso639_5': '',
        'suggested_family_name': '',
        'suggestion_confidence': '',
        'suggestion_reason': '',
        'json_rationale': '',
        'analysis_family_name': '',
        'analysis_mapping_action': '',
        'analysis_review_status': '',
        'analysis_mapping_reason': '',
        'analysis_policy_source': '',
    }
    base.update(row)
    return base


def _build_pair_rows(metadata_dir: Path) -> list[dict[str, object]]:
    recon_path = metadata_dir / 'family_reconciliation.csv'
    if not recon_path.exists():
        raise FileNotFoundError(recon_path)

    recon = _read_csv(recon_path)
    rows = []
    for i, row in recon.iterrows():
        choice = str(row.get('choice', '')).strip()
        n_languages = str(row.get('n_languages', '')).strip()
        review_tier = 'REVIEW_HIGH' if choice == 'anomaly' else 'REFERENCE_ONLY'
        rows.append(_base_review_columns({
            'review_kind': 'iso_glottolog_pair',
            'case_id': f'pair:{i}',
            'review_tier': review_tier,
            'reviewer_decision': choice,
            'reviewer_family_name': row.get('chosen_family_name', ''),
            'n_languages': n_languages,
            'sample_languages': row.get('sample_languages', ''),
            'family_name_iso': row.get('family_name_iso', ''),
            'family_name_glottolog': row.get('family_name_glottolog', ''),
            'choice': choice,
            'category': row.get('category', ''),
            'chosen_family_name': row.get('chosen_family_name', ''),
            'rationale': row.get('rationale', ''),
        }))
    return rows


def _build_json_fallback_review(metadata_dir: Path) -> pd.DataFrame:
    comp_path = metadata_dir / 'language_codes_comprehensive.csv'
    json_path = metadata_dir / 'language_family_assignments.json'
    set5_path = metadata_dir / 'iso_639_set5.csv'
    glottolog_path = metadata_dir / 'glottolog-cache' / 'languoid.csv'

    for path in [comp_path, json_path, set5_path, glottolog_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    comp = pd.read_csv(comp_path, dtype=str, keep_default_na=False)
    with open(json_path, encoding='utf-8') as f:
        json_entries = {entry['language_code']: entry for entry in json.load(f)}
    set5 = pd.read_csv(set5_path, dtype=str, keep_default_na=False)
    cldr_set5_codes = set(set5['iso639_5'].astype(str))
    glottolog_family_by_iso3 = _load_glottolog_family_map(glottolog_path)

    rows = []
    for _, row in comp.iterrows():
        language_code = _code_value(row.get('language_code', ''))
        iso2 = _code_value(row.get('iso639_2_t', ''))
        manual_wins = bool(
            language_code in MANUAL_LANG_TO_SET5
            or (iso2 and iso2 in MANUAL_LANG_TO_SET5)
        )
        entry = json_entries.get(language_code, {})
        json_has_assignment = bool(entry.get('iso639_5') or entry.get('family_name'))
        if manual_wins or not json_has_assignment:
            continue

        iso3 = ''
        for candidate in (language_code, iso2):
            if len(str(candidate)) == 3:
                iso3 = candidate
                break

        raw_family = str(row.get('family_name', '') or '')
        reconciled_family = str(row.get('family_name_reconciled', '') or '')
        glottolog_family = glottolog_family_by_iso3.get(iso3, '') if iso3 else ''
        status = _glottolog_status(raw_family, reconciled_family, glottolog_family)
        has_cldr_hierarchy = str(row.get('iso639_5_family', '') or '') in cldr_set5_codes
        coherence = _suggest_coherent_family(row, entry, status)
        review_tier = _review_tier(status, has_cldr_hierarchy)
        if coherence['coherence_issue'] and coherence['coherence_issue'] != 'no-external-validation':
            review_tier = 'REVIEW_HIGH'

        rows.append({
            'language_code': language_code,
            'language_name': row.get('language_name', ''),
            'sources': row.get('sources', ''),
            'iso639_2_t': iso2,
            'json_iso639_5': entry.get('iso639_5', ''),
            'json_family_name': entry.get('family_name', ''),
            'raw_family_name': raw_family,
            'family_name_reconciled': reconciled_family,
            'glottolog_family': glottolog_family,
            'glottolog_validation_status': status,
            'glottolog_changed_family': raw_family != reconciled_family,
            'has_cldr_iso6395_hierarchy': has_cldr_hierarchy,
            'coherence_issue': coherence['coherence_issue'],
            'suggested_iso639_5': coherence['suggested_iso639_5'],
            'suggested_family_name': coherence['suggested_family_name'],
            'suggestion_confidence': coherence['suggestion_confidence'],
            'suggestion_reason': coherence['suggestion_reason'],
            'review_tier': review_tier,
            'reviewer_decision': '',
            'reviewer_family_name': coherence['suggested_family_name'],
            'reviewer_iso639_5': coherence['suggested_iso639_5'],
            'reviewer_notes': '',
            'json_rationale': entry.get('rationale', ''),
        })

    return pd.DataFrame(rows).sort_values(
        ['review_tier', 'glottolog_validation_status', 'family_name_reconciled', 'language_name'],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def _build_json_rows(metadata_dir: Path, policy: dict[str, dict[str, str]], policy_source: str) -> list[dict[str, object]]:
    json_review = _build_json_fallback_review(metadata_dir)
    rows = []
    for _, row in json_review.iterrows():
        family = str(row.get('family_name_reconciled', '')).strip()
        policy_row = policy.get(family, {})
        review_tier = str(row.get('review_tier', '')).strip()
        if (
            policy_row.get('analysis_review_status') == 'pending'
            or policy_row.get('analysis_mapping_action') == 'row_level_reconciliation'
        ):
            review_tier = 'REVIEW_HIGH'
        reviewer_family_name = str(row.get('reviewer_family_name', '')).strip()
        analysis_family_name = policy_row.get('analysis_family_name', '')
        if (
            not reviewer_family_name
            and analysis_family_name
            and analysis_family_name != family
            and policy_row.get('analysis_mapping_action') in {'normalize_variant', 'exclude_special'}
        ):
            reviewer_family_name = analysis_family_name
        rows.append(_base_review_columns({
            'review_kind': 'json_fallback_language',
            'case_id': f"json:{row.get('language_code', '')}",
            'review_tier': review_tier,
            'reviewer_decision': row.get('reviewer_decision', ''),
            'reviewer_family_name': reviewer_family_name,
            'reviewer_iso639_5': row.get('reviewer_iso639_5', ''),
            'language_code': row.get('language_code', ''),
            'language_name': row.get('language_name', ''),
            'sources': row.get('sources', ''),
            'iso639_2_t': row.get('iso639_2_t', ''),
            'json_iso639_5': row.get('json_iso639_5', ''),
            'json_family_name': row.get('json_family_name', ''),
            'raw_family_name': row.get('raw_family_name', ''),
            'family_name_reconciled': family,
            'glottolog_family': row.get('glottolog_family', ''),
            'glottolog_validation_status': row.get('glottolog_validation_status', ''),
            'glottolog_changed_family': row.get('glottolog_changed_family', ''),
            'has_cldr_iso6395_hierarchy': row.get('has_cldr_iso6395_hierarchy', ''),
            'coherence_issue': row.get('coherence_issue', ''),
            'suggested_iso639_5': row.get('suggested_iso639_5', ''),
            'suggested_family_name': row.get('suggested_family_name', ''),
            'suggestion_confidence': row.get('suggestion_confidence', ''),
            'suggestion_reason': row.get('suggestion_reason', ''),
            'json_rationale': row.get('json_rationale', ''),
            'analysis_family_name': policy_row.get('analysis_family_name', ''),
            'analysis_mapping_action': policy_row.get('analysis_mapping_action', ''),
            'analysis_review_status': policy_row.get('analysis_review_status', ''),
            'analysis_mapping_reason': policy_row.get('analysis_mapping_reason', ''),
            'analysis_policy_source': policy_source,
        }))
    return rows


def build_family_reconciliation_review(
    metadata_dir: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    metadata_dir = Path(metadata_dir)
    policy, policy_source = _load_analysis_policy(metadata_dir)
    rows = _build_pair_rows(metadata_dir) + _build_json_rows(metadata_dir, policy, policy_source)
    out = pd.DataFrame(rows)
    tier_order = {'REVIEW_HIGH': 0, 'REVIEW_MED': 1, 'REFERENCE_ONLY': 2}
    kind_order = {'iso_glottolog_pair': 0, 'json_fallback_language': 1}
    out['_tier_order'] = out['review_tier'].map(tier_order).fillna(9)
    out['_kind_order'] = out['review_kind'].map(kind_order).fillna(9)
    out = (
        out.sort_values(
            ['_tier_order', '_kind_order', 'category', 'family_name_reconciled', 'language_name', 'family_name_iso'],
            kind='stable',
        )
        .drop(columns=['_tier_order', '_kind_order'])
        .reset_index(drop=True)
    )

    output_path = Path(output_path) if output_path else metadata_dir / DEFAULT_OUTPUT
    out.to_csv(output_path, index=False)
    return out


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    table = Table(title='Family Reconciliation Review Queue')
    table.add_column('Review kind')
    table.add_column('Tier')
    table.add_column('Rows', justify='right')
    counts = df.groupby(['review_kind', 'review_tier']).size().reset_index(name='rows')
    for _, row in counts.iterrows():
        table.add_row(row['review_kind'], row['review_tier'], f"{int(row['rows']):,}")
    console.print(table)

    policy_rows = df[
        (
            df['analysis_review_status'].astype(str).str.lower().eq('pending')
            | df['analysis_mapping_action'].astype(str).eq('row_level_reconciliation')
        )
        & df['review_kind'].eq('json_fallback_language')
    ]
    if not policy_rows.empty:
        policy_table = Table(title='JSON Fallback Rows Flagged by Analysis Policy')
        policy_table.add_column('Current family')
        policy_table.add_column('Rows', justify='right')
        for family, n in policy_rows['family_name_reconciled'].value_counts().items():
            policy_table.add_row(str(family), f'{int(n):,}')
        console.print(policy_table)

    console.print(f'[green]Wrote:[/green] {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-dir', default=None)
    parser.add_argument('--output-path', default=None)
    args = parser.parse_args()

    data_dir = Path(get_data_directory_path())
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else data_dir / 'metadata_files'
    output_path = Path(args.output_path) if args.output_path else metadata_dir / DEFAULT_OUTPUT
    df = build_family_reconciliation_review(metadata_dir, output_path)
    print_summary(df, output_path)


if __name__ == '__main__':
    main()
