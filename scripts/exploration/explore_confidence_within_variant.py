#!/usr/bin/env python3
"""
explore_confidence_within_variant.py
======================================
Frequency-based agreement scoring across translation services WITHIN each prompt variant.

For each prompt variant separately, score agreement both among the eight LLM
services (OpenAI, Claude, Gemini, DeepSeek, Llama, Gemma, Qwen, Mistral) and
among the four prompt-invariant non-LLM sources — three MT baselines (Google
Translate, EasyNMT, Lingvanex) plus the Wikipedia community reference. The
non-LLM block is a sanity check and a reference anchor; the LLM block is the
primary signal of interest.

This explores:
  - Within a single variant, which term/language combos have high LLM agreement?
  - Do API models (OpenAI, Claude, Gemini, DeepSeek) agree more than local
    models (Llama, Gemma, Qwen, Mistral), or vice versa?
  - Which rows are strong auto-approve candidates vs. need human review?

Compare with: explore_confidence_across_variants.py (which measures if different prompts produce the same translations).

Adapts the FrequencyConfidenceCalculator approach from Kraus et al. (2025, WOKIE) but preserves the full candidate distribution, so disagreements are logged as research data.

Output files are written to data_directory/translated_terms/{term}/evaluation/:
  - `confidence_scores.csv`: all rows with confidence scores
  - `confidence_summary.csv`: aggregate stats per term/variant

Usage:
    python explore_confidence_within_variant.py
    python explore_confidence_within_variant.py --term "Computational Humanities"
    python explore_confidence_within_variant.py --variants github_searcher minimal
    python explore_confidence_within_variant.py --output-dir /path/to/out
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: F401

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.utils import get_data_directory_path, read_csv_file, analyze_differences

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

# Prompt-invariant services — same across all variant CSVs.
# Used as a fixed reference baseline, NOT as the within-variant scoring pool.
BASELINE_SERVICES = {
    'gt':        'gt_translated_term',
    'enmt':      'enmt_translated_term',
    'lingvanex': 'lingvanex_translated_term',
    'wikipedia': 'wikipedia_translated_term',
}

# LLM services — these actually ran with different prompts per variant, so they
# are the correct pool for measuring within-variant agreement.
LLM_SERVICES = {
    'openai':   'openai_translated_term',
    'claude':   'claude_translated_term',
    'gemini':   'gemini_translated_term',
    'deepseek': 'deepseek_translated_term',
    'llama':    'llama_translated_term',
    'gemma':    'gemma_translated_term',
    'qwen':     'qwen_translated_term',
    'mistral':  'mistral_translated_term',
}

# Keep legacy alias so callers that reference PRIMARY_SERVICES still work
PRIMARY_SERVICES = BASELINE_SERVICES

ALL_VARIANTS = [
    'minimal', 'fluent_speaker', 'github_searcher', 'judge'
]


# ── Core scoring logic ─────────────────────────────────────────────────────────

def score_row_confidence(
    row: pd.Series,
    service_cols: Dict[str, str],
) -> dict:
    """
    Compute frequency-based confidence for a single translation row.

    Parameters
    ----------
    row : pd.Series
    service_cols : dict
        Mapping of {service_name: column_name} to include in scoring.

    Returns
    -------
    dict with keys:
        best_candidate, confidence, total_services, agreeing_services,
        candidate_distribution, unique_candidates
    """
    counts: Dict[str, int] = defaultdict(int)
    service_values: Dict[str, Optional[str]] = {}
    total = 0

    for service, col in service_cols.items():
        val = row.get(col)
        if pd.notna(val) and isinstance(val, str) and len(val) > 0:
            service_values[service] = val
            counts[val] += 1
            total += 1
        else:
            service_values[service] = None

    if total == 0:
        return {
            'best_candidate': None,
            'confidence': 0.0,
            'total_services': 0,
            'agreeing_services': 0,
            'candidate_distribution': '{}',
            'unique_candidates': 0,
            'service_values': str(service_values),
        }

    best_candidate = max(counts, key=counts.get)
    confidence = counts[best_candidate] / total
    unique_candidates = len(counts)

    # Candidate distribution: {term: count} sorted by frequency
    dist = {k: v for k, v in sorted(counts.items(), key=lambda x: -x[1])}

    # Analyze what's causing differences (using language directionality if available)
    directionality = row.get('directionality', 'ltr') or 'ltr'
    differences = analyze_differences(service_values, directionality)

    return {
        'best_candidate': best_candidate,
        'confidence': round(confidence, 4),
        'total_services': total,
        'agreeing_services': counts[best_candidate],
        'candidate_distribution': str(dist),
        'unique_candidates': unique_candidates,
        'service_values': str(service_values),
        'difference_types': differences.get('summary', 'unknown'),
        'difference_details': str(differences),
    }


def load_variant_df(
    data_directory_path: str,
    term_slug: str,
    variant: str,
) -> Optional[pd.DataFrame]:
    """
    Build the full per-variant DataFrame by merging per-service files.

    Loads all direct_services/ and prompt_services/ files whose names contain
    the variant name (for prompt services) or are variant-invariant (for direct
    services), then merges them on (language_code, term_source).

    Before returning, ``enforce_translation_rationale_pairing`` is applied so
    that every LLM service column pair (``{svc}_translated_term`` /
    ``{svc}_translation_rationale``) is internally consistent:
    - A translation with no real rationale (absent or placeholder such as
      "No rationale provided") has its translation cell set to NaN.
    - A rationale with no translation has its rationale cell set to NaN.
    This means all downstream callers — notebooks and scripts — receive clean
    data without needing to handle these mismatches themselves.

    Parameters
    ----------
    data_directory_path : str
    term_slug : str
    variant : str

    Returns
    -------
    pd.DataFrame with all available services merged together for this
    term/variant, or None if no data found.
    """
    term_dir = os.path.join(
        data_directory_path, 'translated_terms', term_slug
    )
    # Stable join key — present in every per-service file including Wikipedia
    join_cols = ['language_code', 'term_source']

    def _load(path: str) -> Optional[pd.DataFrame]:
        try:
            return read_csv_file(path)
        except Exception as e:
            console.print(f"  ⚠ Could not load {path}: {e}", style="dim yellow")
            return None

    # Direct services are prompt-invariant — load all of them
    direct_dir = os.path.join(term_dir, 'direct_services')
    frames = []
    if os.path.isdir(direct_dir):
        for fname in sorted(os.listdir(direct_dir)):
            if fname.endswith('.csv'):
                df = _load(os.path.join(direct_dir, fname))
                if df is not None:
                    frames.append(df)

    # Prompt services — load only files that match this variant
    prompt_dir = os.path.join(term_dir, 'prompt_services')
    if os.path.isdir(prompt_dir):
        for fname in sorted(os.listdir(prompt_dir)):
            if fname.endswith('.csv') and variant in fname:
                df = _load(os.path.join(prompt_dir, fname))
                if df is not None:
                    frames.append(df)

    if not frames:
        return None

    # Merge all frames together on the stable key. Per-service CSVs carry all prior pipeline columns (including NaN stubs for services that hadn't run yet when they were saved). The standalone service files (e.g. wikipedia_translations.csv) are authoritative for their own column, so we also fill NaN values in already-present columns from those files.
    base = frames[0]
    for other in frames[1:]:
        merge_on = [c for c in join_cols if c in other.columns]
        new_cols = [c for c in other.columns if c not in base.columns]
        # Columns that already exist in base but are NaN — fill from this file
        fillable = [
            c for c in other.columns
            if c in base.columns and c not in merge_on and base[c].isna().any()
        ]
        if not new_cols and not fillable:
            continue
        merged = base.merge(
            other[merge_on + new_cols + fillable],
            on=merge_on, how='outer', suffixes=('', '_fill'),
        )
        for c in fillable:
            fill_col = f'{c}_fill'
            if fill_col in merged.columns:
                merged[c] = merged[c].fillna(merged[fill_col])
                merged = merged.drop(columns=[fill_col])
        base = merged

    # Keep most recent row per language (in case of duplicates from outer merges)
    if 'coding_dh_date' in base.columns:
        base['coding_dh_date'] = pd.to_datetime(base['coding_dh_date'], errors='coerce')
        base = base.sort_values('coding_dh_date', ascending=False)
    base = base.drop_duplicates(subset=['language_code', 'term_source'], keep='first')
    base['prompt_variant'] = variant
    from scripts.utils import enforce_translation_rationale_pairing
    base = enforce_translation_rationale_pairing(base)
    return base.reset_index(drop=True)


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add confidence scoring columns to a translations DataFrame.

    Scoring pools:
    - ``llm_*``: LLM services only (OpenAI, Claude, Gemini, Ollama). These vary per prompt variant and are the main signal.
    - ``baseline_*``: Non-LLM services (GT, EasyNMT, Lingvanex, Wikipedia). Prompt-invariant — identical across all variant CSVs, kept as a fixed reference for comparison.
    - ``all_*`` : All services combined.
    
    Parameters
    ----------
    df : pd.DataFrame
		DataFrame containing translation outputs from various services for a single term/variant. Must include columns corresponding to the services in LLM_SERVICES and BASELINE_SERVICES.
          
    Returns
    -------
    pd.DataFrame with new columns appended for confidence scores and related metadata.
    """
    # LLM-only confidence — main scoring pool (varies per variant)
    llm_pool = dict(LLM_SERVICES)
    available_llm = {k: v for k, v in llm_pool.items() if v in df.columns}
    llm_scores = df.apply(lambda row: score_row_confidence(row, available_llm), axis=1)
    llm_df = pd.DataFrame(llm_scores.tolist())
    llm_df.columns = [f'llm_{c}' for c in llm_df.columns]

    # Baseline service confidence (prompt-invariant reference)
    baseline_scores = df.apply(lambda row: score_row_confidence(row, BASELINE_SERVICES), axis=1)
    baseline_df = pd.DataFrame(baseline_scores.tolist())
    baseline_df.columns = [f'baseline_{c}' for c in baseline_df.columns]

    # All-service confidence (LLM + baseline combined)
    all_services = {**BASELINE_SERVICES, **LLM_SERVICES}
    available_all = {k: v for k, v in all_services.items() if v in df.columns}
    all_scores = df.apply(lambda row: score_row_confidence(row, available_all), axis=1)
    all_df = pd.DataFrame(all_scores.tolist())
    all_df.columns = [f'all_{c}' for c in all_df.columns]

    return pd.concat([df.reset_index(drop=True), llm_df, baseline_df, all_df], axis=1)


def compute_summary(scored_df: pd.DataFrame, variant: str, term: str) -> dict:
    """
    Compute aggregate stats for a single variant/term combination.
     
    Parameters
    ----------
    scored_df : pd.DataFrame
    	DataFrame with confidence scores computed for a single term/variant.
	variant : str
		Prompt variant name (e.g. 'github_searcher', 'minimal', etc.) — used for labeling the summary row.
	term : str
     	Term name (e.g. 'Digital Humanities') — used for labeling the summary row. 

    Returns
    -------
    dict with keys:
    - term
    - variant
    - total_languages
    - languages_with_llm
    - mean_llm_confidence
    - median_llm_confidence
    - std_llm_confidence
    - cv_llm_confidence
    - llm_unique_candidates_mean
    - languages_with_baseline
    - mean_baseline_confidence
     
    """
    total = len(scored_df)
    has_llm = scored_df['llm_total_services'] > 0
    has_baseline = scored_df['baseline_total_services'] > 0

    return {
        'term': term,
        'variant': variant,
        'total_languages': total,
        # LLM agreement (main signal — varies per variant)
        'languages_with_llm': int(has_llm.sum()),
        'mean_llm_confidence': round(scored_df.loc[has_llm, 'llm_confidence'].mean(), 4) if has_llm.any() else None,
        'median_llm_confidence': round(scored_df.loc[has_llm, 'llm_confidence'].median(), 4) if has_llm.any() else None,
        'std_llm_confidence': round(scored_df.loc[has_llm, 'llm_confidence'].std(), 4) if has_llm.any() else None,
        'cv_llm_confidence': round(
            scored_df.loc[has_llm, 'llm_confidence'].std() / scored_df.loc[has_llm, 'llm_confidence'].mean(), 4
        ) if has_llm.any() and scored_df.loc[has_llm, 'llm_confidence'].mean() > 0 else None,
        'llm_unique_candidates_mean': round(scored_df.loc[has_llm, 'llm_unique_candidates'].mean(), 2) if has_llm.any() else None,
        # Baseline reference (prompt-invariant — should be same across all variants)
        'languages_with_baseline': int(has_baseline.sum()),
        'mean_baseline_confidence': round(scored_df.loc[has_baseline, 'baseline_confidence'].mean(), 4) if has_baseline.any() else None,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_confidence_evaluation(
    data_directory_path: str,
    target_terms: List[str],
    variants: List[str] = ALL_VARIANTS,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run confidence scoring for all term/variant combinations.
    
    Parameters
    ----------
    data_directory_path : str
		Base path to the data directory containing translated_terms/
    target_terms : list of str
		Terms to evaluate (e.g. ['Digital Humanities'])
    variants : list of str
		Prompt variants to include (e.g. ['github_searcher', 'minimal', etc.])
	output_dir : str, optional
		Optional output directory. If not provided, defaults to data_directory_path/translated_terms/{term}/evaluation/

    Returns
    -------
    scored_df : pd.DataFrame
        All rows with confidence scores appended.
    summary_df : pd.DataFrame
        One row per term/variant with aggregate stats.
    """
    all_scored = []
    summaries = []

    for term in target_terms:
        term_slug = term.lower().replace(' ', '_')
        console.print(f"\n📊 Scoring: {term}", style="bold cyan")

        for variant in variants:
            df = load_variant_df(data_directory_path, term_slug, variant)
            if df is None:
                console.print(f"  ⚠ No data for variant '{variant}' — skipping", style="dim yellow")
                continue

            console.print(f"  ✓ Loaded {len(df)} rows for variant '{variant}'", style="green")
            scored = score_dataframe(df)
            all_scored.append(scored)
            summaries.append(compute_summary(scored, variant, term))

    if not all_scored:
        console.print("⚠ No data found for any term/variant combination.", style="bold red")
        return pd.DataFrame(), pd.DataFrame()

    scored_df = pd.concat(all_scored, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    # Save outputs
    _dir = output_dir or os.path.join(
        data_directory_path, 'translated_terms', target_terms[0].lower().replace(' ', '_'), 'evaluation'
    )
    os.makedirs(_dir, exist_ok=True)

    scored_df.to_csv(os.path.join(_dir, 'confidence_scores.csv'), index=False)
    summary_df.to_csv(os.path.join(_dir, 'confidence_summary.csv'), index=False)

    console.print(f"\n✓ Outputs written to: {_dir}", style="bold green")
    console.print(f"  confidence_scores.csv  : {len(scored_df)} rows", style="green")
    console.print(f"  confidence_summary.csv : {len(summary_df)} rows", style="green")

    _print_summary_table(summary_df)

    return scored_df, summary_df


def _print_summary_table(summary_df: pd.DataFrame) -> None:
    cols = [
        'term', 'variant', 'total_languages',
        'languages_with_llm', 'mean_llm_confidence',
        'median_llm_confidence', 'std_llm_confidence', 'cv_llm_confidence',
        'llm_unique_candidates_mean',
    ]
    table = Table(title="Confidence Scoring Summary (LLM agreement per variant)", show_header=True)
    for col in cols:
        table.add_column(col, style="cyan" if col in ('term', 'variant') else "white")
    for _, row in summary_df.iterrows():
        table.add_row(*[str(row.get(c, '')) for c in cols])
    console.print(table)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score translation agreement across primary services."
    )
    parser.add_argument(
        '--term', nargs='+',
        default=['Digital Humanities'],
        help='Target terms to evaluate'
    )
    parser.add_argument(
        '--variants', nargs='+', default=ALL_VARIANTS,
        choices=ALL_VARIANTS,
        help='Prompt variants to include'
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory (default: data_dir/translated_terms/{term}/evaluation/)'
    )
    args = parser.parse_args()

    data_dir = get_data_directory_path()
    run_confidence_evaluation(
        data_directory_path=data_dir,
        target_terms=args.term,
        variants=args.variants,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()