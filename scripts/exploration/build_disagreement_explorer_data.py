#!/usr/bin/env python3
"""
build_disagreement_explorer_data.py
====================================
Merge disagreement analysis, per-language confidence stats, rationale lengths,
and parsed service translations into one notebook-support CSV.

Before writing, ``enforce_translation_rationale_pairing`` (scripts/utils.py) is
applied to each per-variant slice of confidence_scores.csv. This nulls out any
LLM cell where the translation has no real rationale (absent or placeholder such
as "No rationale provided") or vice versa, so notebook 07 never receives
unpaired cells regardless of whether the upstream CSVs were generated before or
after this enforcement was introduced.

Run after:
  1. scripts/exploration/explore_confidence_within_variant.py
     (or notebook 04, which writes confidence_scores.csv)
  2. scripts/exploration/explore_disagreements.py
     (or notebook 05, which writes disagreement_analysis.csv)

Usage
-----
    python scripts/exploration/build_disagreement_explorer_data.py
    python scripts/exploration/build_disagreement_explorer_data.py --term "Digital Humanities"
    python scripts/exploration/build_disagreement_explorer_data.py --output-dir path/to/dir
"""

import argparse
import ast
import os
import sys

import pandas as pd

RATIONALE_VARIANT = 'fluent_speaker'

VARIANTS = ['minimal', 'fluent_speaker', 'github_searcher', 'judge']

RATIONALE_SERVICE_COLS = {
    'claude':   'claude_translation_rationale',
    'openai':   'openai_translation_rationale',
    'gemini':   'gemini_translation_rationale',
    'deepseek': 'deepseek_translation_rationale',
    'llama':    'llama_translation_rationale',
    'gemma':    'gemma_translation_rationale',
    'qwen':     'qwen_translation_rationale',
    'mistral':  'mistral_translation_rationale',
}

TERM_SERVICE_COLS = {
    'claude':   'claude_translated_term',
    'openai':   'openai_translated_term',
    'gemini':   'gemini_translated_term',
    'deepseek': 'deepseek_translated_term',
    'llama':    'llama_translated_term',
    'gemma':    'gemma_translated_term',
    'qwen':     'qwen_translated_term',
    'mistral':  'mistral_translated_term',
}

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils import get_data_directory_path, enforce_translation_rationale_pairing

CAT_ORDER = {
    'TRANSMOGRIFICATION': 0,
    'STRUCTURAL_ABSENCE': 1,
    'PRODUCTIVE_DISAGREEMENT': 2,
    'MEASUREMENT_ARTEFACT': 3,
    'COMPLETE_CONSENSUS': 4,
}

SERVICE_MAP = {
    'Google Translate': 'gt_term',
    'OpenAI':           'openai_term',
    'Claude':           'claude_term',
    'Gemini':           'gemini_term',
    'DeepSeek':         'deepseek_term',
    'Llama':            'llama_term',
    'Gemma':            'gemma_term',
    'Qwen':             'qwen_term',
    'Mistral':          'mistral_term',
}


def build_explorer_data(
    data_directory_path: str,
    term: str = 'Digital Humanities',
    output_dir: str | None = None,
    rationale_variant: str = RATIONALE_VARIANT,
) -> pd.DataFrame:
    eval_dir = os.path.join(
        data_directory_path, 'translated_terms',
        term.lower().replace(' ', '_'), 'evaluation'
    )

    disagr_path = os.path.join(eval_dir, 'disagreement_analysis.csv')
    conf_path = os.path.join(eval_dir, 'confidence_scores.csv')

    if not os.path.exists(disagr_path):
        sys.exit(f"Missing: {disagr_path}\nRun explore_disagreements.py first.")
    if not os.path.exists(conf_path):
        sys.exit(f"Missing: {conf_path}\nRun the confidence scoring notebook first.")

    _conv = {'language_code': str}
    base = pd.read_csv(disagr_path, converters=_conv)
    conf_df = pd.read_csv(conf_path, converters=_conv)

    # ── Confidence stats: aggregate across all prompt variants per language ──
    conf_agg = (
        conf_df
        .groupby('language_code')
        .agg(
            mean_llm_confidence=('llm_confidence', 'mean'),
            std_llm_confidence=('llm_confidence', 'std'),
            mean_all_confidence=('all_confidence', 'mean'),
            mean_baseline_confidence=('baseline_confidence', 'mean'),
            mean_agreeing_services=('llm_agreeing_services', 'mean'),
            n_total_services=('llm_total_services', 'first'),
        )
        .reset_index()
    )

    # ── Mean rationale length: average character count of rationale snippets ─
    rationale_cols = [c for c in base.columns if c.endswith('_rationale_snippet')]
    if rationale_cols:
        base['mean_rationale_length'] = (
            base[rationale_cols]
            .apply(lambda row: row.dropna().map(lambda x: len(str(x)) if x not in ('nan', 'None', '') else 0), axis=1)
            .mean(axis=1)
            .round(1)
        )

    # ── Parse service_translations dict into individual columns ─────────────
    parsed_rows = []
    for _, row in base.iterrows():
        d = {'language_code': row['language_code']}
        try:
            svc = ast.literal_eval(str(row['service_translations']))
            for svc_name, col_name in SERVICE_MAP.items():
                d[col_name] = svc.get(svc_name, '')
        except Exception:
            for col_name in SERVICE_MAP.values():
                d[col_name] = ''
        parsed_rows.append(d)
    parsed_df = pd.DataFrame(parsed_rows)

    # ── Terms + rationales for every variant × service combination ──────────
    # Produces columns like claude_term_minimal, claude_rationale_judge, etc.
    matrix_frames = []
    for variant in VARIANTS:
        variant_rows = conf_df[conf_df['prompt_variant'] == variant].copy()
        if variant_rows.empty:
            continue
        # Null out unpaired LLM translation/rationale cells before they reach the explorer.
        # confidence_scores.csv may have been written before pairing enforcement was added
        # to load_variant_df, so we re-apply it here to guarantee clean output.
        variant_rows = enforce_translation_rationale_pairing(variant_rows)
        avail_rat = {svc: col for svc, col in RATIONALE_SERVICE_COLS.items() if col in conf_df.columns}
        avail_term = {svc: col for svc, col in TERM_SERVICE_COLS.items() if col in conf_df.columns}
        src_cols = list(set(avail_rat.values()) | set(avail_term.values()))
        rename_map = (
            {col: f'{svc}_rationale_{variant}' for svc, col in avail_rat.items()}
            | {col: f'{svc}_term_{variant}' for svc, col in avail_term.items()}
        )
        frame = (
            variant_rows[['language_code'] + src_cols]
            .drop_duplicates('language_code')
            .rename(columns=rename_map)
        )
        matrix_frames.append(frame)

    if matrix_frames:
        rat_df = matrix_frames[0]
        for frame in matrix_frames[1:]:
            rat_df = rat_df.merge(frame, on='language_code', how='outer')
    else:
        rat_df = pd.DataFrame(columns=['language_code'])

    # ── Baseline terms (non-LLM services, don't vary by prompt variant) ────
    baseline_cols = ['wikipedia_translated_term', 'lingvanex_translated_term', 'enmt_translated_term']
    available_baseline = [c for c in baseline_cols if c in conf_df.columns]
    wiki_df = (
        conf_df[['language_code'] + available_baseline]
        .drop_duplicates('language_code')
        .loc[lambda d: d['language_code'].str.strip() != '']
    )

    # ── Merge ────────────────────────────────────────────────────────────────
    export = (
        base
        .merge(conf_agg, on='language_code', how='left')
        .merge(parsed_df, on='language_code', how='left')
        .merge(rat_df, on='language_code', how='left')
        .merge(wiki_df, on='language_code', how='left')
    )

    # ── Sort by most disagreement first ─────────────────────────────────────
    export['_cat_order'] = export['category'].map(CAT_ORDER).fillna(9)
    export = (
        export
        .sort_values(['_cat_order', 'n_unique_raw'], ascending=[True, False])
        .drop(columns=['_cat_order'])
        .reset_index(drop=True)
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = output_dir or eval_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'disagreement_explorer_data.csv')
    export.to_csv(out_path, index=False)
    print(f"Saved {len(export)} rows × {len(export.columns)} cols → {out_path}")

    cat_counts = export['category'].value_counts()
    for cat, n in cat_counts.items():
        print(f"  {cat}: {n}")

    return export


def main():
    parser = argparse.ArgumentParser(
        description='Build disagreement_explorer_data.csv for notebook support.'
    )
    parser.add_argument('--term', default='Digital Humanities', help='Target term (default: Digital Humanities)')
    parser.add_argument('--output-dir', default=None, help='Override output directory')
    parser.add_argument(
        '--rationale-variant', default=RATIONALE_VARIANT,
        choices=['minimal', 'fluent_speaker', 'github_searcher', 'judge'],
        help=f'Prompt variant to pull full rationales from (default: {RATIONALE_VARIANT})'
    )
    args = parser.parse_args()

    data_dir = get_data_directory_path()
    build_explorer_data(
        data_directory_path=data_dir,
        term=args.term,
        output_dir=args.output_dir,
        rationale_variant=args.rationale_variant,
    )


if __name__ == '__main__':
    main()
