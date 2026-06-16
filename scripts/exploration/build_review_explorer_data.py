#!/usr/bin/env python3
"""
build_review_explorer_data.py
==============================
Build review_explorer_data.csv for the HTML review explorer.

Reads raw translations (prompt_services + direct_services) and
automated_review_signals.csv produced by notebook 02. No dependency on
notebooks 03–07.

The output has one row per language and contains:
  - automated review signal columns (from automated_review_signals.csv)
  - review_tier: REVIEW_HIGH (≥2 flags) / REVIEW_MED (1 flag) / CLEAN (0 flags)
  - {svc}_term_{variant} / {svc}_rationale_{variant} for 8 LLMs × 4 variants
    API services: claude, openai, gemini, deepseek
    Local services: llama, gemma, qwen, mistral
  - baseline terms: wikipedia_translated_term, gt_translated_term,
                    enmt_translated_term, lingvanex_translated_term

Usage
-----
    python scripts/exploration/build_review_explorer_data.py
    python scripts/exploration/build_review_explorer_data.py --term "Digital Humanities"
    python scripts/exploration/build_review_explorer_data.py --output-dir path/to/dir
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils import get_data_directory_path, enforce_translation_rationale_pairing
from scripts.exploration.explore_confidence_within_variant import load_variant_df
from scripts.exploration.translation_classifier import curate_translation

VARIANTS = ['minimal', 'fluent_speaker', 'github_searcher', 'judge']

LLM_TERM_COLS = {
    'claude':   'claude_translated_term',
    'openai':   'openai_translated_term',
    'gemini':   'gemini_translated_term',
    'deepseek': 'deepseek_translated_term',
    'llama':    'llama_translated_term',
    'gemma':    'gemma_translated_term',
    'qwen':     'qwen_translated_term',
    'mistral':  'mistral_translated_term',
}
LLM_RAT_COLS = {
    'claude':   'claude_translation_rationale',
    'openai':   'openai_translation_rationale',
    'gemini':   'gemini_translation_rationale',
    'deepseek': 'deepseek_translation_rationale',
    'llama':    'llama_translation_rationale',
    'gemma':    'gemma_translation_rationale',
    'qwen':     'qwen_translation_rationale',
    'mistral':  'mistral_translation_rationale',
}
BASELINE_COLS = [
    'wikipedia_translated_term',
    'gt_translated_term',
    'enmt_translated_term',
    'lingvanex_translated_term',
]

_TIER_ORDER = {'REVIEW_HIGH': 0, 'REVIEW_MED': 1, 'CLEAN': 2}


def _assign_tier(flag_count: int) -> str:
    if flag_count >= 2:
        return 'REVIEW_HIGH'
    if flag_count == 1:
        return 'REVIEW_MED'
    return 'CLEAN'


def build_review_data(
    data_directory_path: str,
    term: str = 'Digital Humanities',
    output_dir: str | None = None,
) -> pd.DataFrame:
    term_slug = term.lower().replace(' ', '_')
    eval_dir = os.path.join(data_directory_path, 'translated_terms', term_slug, 'evaluation')
    flags_path = os.path.join(eval_dir, 'automated_review_signals.csv')

    if not os.path.exists(flags_path):
        sys.exit(
            f"Missing: {flags_path}\n"
            "Run notebook 02 first (Automated Review Signals section)."
        )

    flags_df = pd.read_csv(flags_path, converters={'language_code': str})

    # ── Load all variants, build per-language wide matrix ────────────────────
    baseline_df: pd.DataFrame | None = None
    matrix_frames: list[pd.DataFrame] = []

    for variant in VARIANTS:
        vdf = load_variant_df(data_directory_path, term_slug, variant)
        if vdf is None or vdf.empty:
            continue

        # Apply pairing enforcement so explorer never sees unpaired cells
        vdf = enforce_translation_rationale_pairing(vdf)

        # One row per language (first occurrence wins)
        vdf = vdf.drop_duplicates('language_code')

        # Rename LLM columns to variant-suffixed forms
        rename_map: dict[str, str] = {}
        for svc, src_col in LLM_TERM_COLS.items():
            if src_col in vdf.columns:
                rename_map[src_col] = f'{svc}_term_{variant}'
        for svc, src_col in LLM_RAT_COLS.items():
            if src_col in vdf.columns:
                rename_map[src_col] = f'{svc}_rationale_{variant}'

        llm_renamed_cols = list(rename_map.values())
        frame = (
            vdf[['language_code'] + [c for c in rename_map if c in vdf.columns]]
            .rename(columns=rename_map)
        )
        matrix_frames.append(frame)

        # Baseline columns + translation date: read once from the minimal variant
        if variant == 'minimal' and baseline_df is None:
            avail_baseline = [c for c in BASELINE_COLS if c in vdf.columns]
            baseline_df = vdf[['language_code'] + avail_baseline].copy()
            if 'coding_dh_date' in vdf.columns:
                dates = pd.to_datetime(vdf['coding_dh_date'], errors='coerce').dt.date.astype(str)
                baseline_df['translation_date'] = dates.values

    if not matrix_frames:
        sys.exit('No translation data found for any variant.')

    # Merge all variant frames
    term_rat_df = matrix_frames[0]
    for frame in matrix_frames[1:]:
        new_cols = [c for c in frame.columns if c != 'language_code']
        term_rat_df = term_rat_df.merge(
            frame[['language_code'] + new_cols], on='language_code', how='outer'
        )

    # ── Merge everything ──────────────────────────────────────────────────────
    export = flags_df.copy()

    if baseline_df is not None and not baseline_df.empty:
        export = export.merge(baseline_df, on='language_code', how='left')

    llm_cols = [c for c in term_rat_df.columns if c != 'language_code']
    export = export.merge(
        term_rat_df[['language_code'] + llm_cols], on='language_code', how='left'
    )

    # ── Add clean action + cleaned term columns ───────────────────────────────
    # Both the action ('stripped'/'nulled'/'placeholder'/'unchanged') and the
    # cleaned result are stored so the HTML explorer can display the exact
    # proposed term without reimplementing classifier logic in JavaScript.
    for variant in VARIANTS:
        for svc in LLM_TERM_COLS.keys():
            term_col   = f'{svc}_term_{variant}'
            action_col = f'{svc}_clean_action_{variant}'
            clean_col  = f'{svc}_term_clean_{variant}'
            if term_col not in export.columns:
                continue

            def _classify(t):
                if isinstance(t, str) and t.strip() and t not in ('nan', 'None'):
                    return curate_translation(t)
                return t, 'unchanged'

            pairs = export[term_col].apply(_classify)
            export[action_col] = pairs.apply(lambda p: p[1])
            export[clean_col]  = pairs.apply(lambda p: p[0] if p[1] == 'stripped' else None)

    # ── Unpaired term columns (term present in raw CSV but nulled by pairing) ──
    # When a service produced a term but no rationale, enforce_translation_rationale_pairing
    # nulls the term. We store the original here so the HTML explorer can show it
    # when it matches another service's paired term (convergence signal without exclusion).
    prompt_dir = os.path.join(data_directory_path, 'translated_terms', term_slug, 'prompt_services')
    for variant in VARIANTS:
        for svc, term_col in LLM_TERM_COLS.items():
            paired_col   = f'{svc}_term_{variant}'
            unpaired_col = f'{svc}_term_unpaired_{variant}'
            raw_path = os.path.join(prompt_dir, f'{svc}_{variant}_translations.csv')
            if not os.path.exists(raw_path) or paired_col not in export.columns:
                continue
            try:
                raw_df = pd.read_csv(raw_path, usecols=['language_code', term_col], converters={'language_code': str})
            except (ValueError, KeyError):
                continue
            raw_df = raw_df.drop_duplicates('language_code')
            merged = export[['language_code', paired_col]].merge(
                raw_df.rename(columns={term_col: '_raw'}),
                on='language_code', how='left',
            )
            paired_null = merged[paired_col].isna() | merged[paired_col].astype(str).str.strip().isin(['', 'nan', 'None'])
            raw_valid   = merged['_raw'].notna() & ~merged['_raw'].astype(str).str.strip().isin(['', 'nan', 'None'])
            export[unpaired_col] = merged['_raw'].where(paired_null & raw_valid).values

    # ── Review tier ───────────────────────────────────────────────────────────
    export['review_tier'] = export['flag_count'].apply(_assign_tier)

    # ── Auto-exclusion hints for HTML review interface ────────────────────────
    # Keep this aligned with scripts.utils.filter_for_analysis():
    # xall  → likely term error (excluded in quality and search_ready modes)
    # xsrch → likely search complication (excluded in search_ready mode only)
    _AUTO_XALL_FLAGS = {
        'has_repetition_loop',
        'has_mixed_script',
        'has_placeholder_term',
        'has_unicode_escape',
        'has_extreme_term_length',
    }
    _AUTO_XSRCH_FLAGS = {
        'has_romanization',
        'has_source_term',
        'has_short_translation',
    }

    def _auto_exclusion_hint(row):
        def flag(col):
            v = row.get(col)
            return v is True or str(v).strip().lower() == 'true'
        if any(flag(f) for f in _AUTO_XALL_FLAGS if f in row.index):
            return 'xall'
        if any(flag(f) for f in _AUTO_XSRCH_FLAGS if f in row.index):
            return 'xsrch'
        return ''

    export['auto_lang_exclusion'] = export.apply(_auto_exclusion_hint, axis=1)

    # ── Sort: REVIEW_HIGH → REVIEW_MED → CLEAN; within tier by flag_count desc
    export['_tier_order'] = export['review_tier'].map(_TIER_ORDER).fillna(9)
    export = (
        export
        .sort_values(['_tier_order', 'flag_count', 'language_name'], ascending=[True, False, True])
        .drop(columns=['_tier_order'])
        .reset_index(drop=True)
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = output_dir or eval_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'review_explorer_data.csv')
    export.to_csv(out_path, index=False)
    print(f"Saved {len(export)} rows × {len(export.columns)} cols → {out_path}")

    for tier in ['REVIEW_HIGH', 'REVIEW_MED', 'CLEAN']:
        n = (export['review_tier'] == tier).sum()
        print(f"  {tier}: {n}")

    return export


def main():
    parser = argparse.ArgumentParser(
        description='Build review_explorer_data.csv for the HTML review explorer.'
    )
    parser.add_argument(
        '--term', default='Digital Humanities',
        help='Target term (default: Digital Humanities)',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Override output directory (default: evaluation/ dir for the term)',
    )
    args = parser.parse_args()

    data_dir = get_data_directory_path()
    build_review_data(
        data_directory_path=data_dir,
        term=args.term,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
