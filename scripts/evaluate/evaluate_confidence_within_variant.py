#!/usr/bin/env python3
"""
evaluate_confidence_within_variant.py
======================================
Level 0 evaluation: Frequency-based agreement scoring across translation services
WITHIN each prompt variant.

For each prompt variant separately, measure how much primary translation services
(Google Translate, EasyNMT, Lingvanex, Wikipedia) agree on the same translation.

This explores:
  - Within a single variant, which term/language combos have high service agreement?
  - How consistent are independent translation engines for each prompt variant?
  - Which rows are strong auto-approve candidates vs. need human review?

Compare with: evaluate_confidence_across_variants.py (which measures if different
prompts produce the same translations).

Adapts the FrequencyConfidenceCalculator approach from Kraus et al. (2025, WOKIE)
— but preserves the full candidate distribution, so disagreements are logged as
research data.

Output files are written to data_directory/metadata_files/evaluation/:
  - `confidence_scores.csv`: all rows with confidence scores
  - `confidence_summary.csv`: aggregate stats per term/variant
  - `low_confidence.csv`: rows below threshold (potential hallucinations)

Usage:
    python evaluate_confidence_within_variant.py
    python evaluate_confidence_within_variant.py --term "Computational Humanities"
    python evaluate_confidence_within_variant.py --variants comparative minimal
    python evaluate_confidence_within_variant.py --threshold 0.6 --output-dir /path/to/out
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import get_data_directory_path, read_csv_file

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────

# Primary (non-LLM) services — these form the agreement pool for confidence
PRIMARY_SERVICES = {
    'gt':        'gt_translated_term',
    'enmt':      'enmt_translated_term',
    'lingvanex': 'lingvanex_translated_term',
    'wikipedia': 'wikipedia_translated_term',
}

# LLM services — scored separately, not part of primary agreement
LLM_SERVICES = {
    'openai':  'openai_translated_term',
    'claude':  'claude_translated_term',
    'gemini':  'gemini_translated_term',
    'ollama':  'ollama_translated_term',
}

ALL_VARIANTS = [
    'comparative', 'minimal', 'expert_persona', 'contextual', 'native_rationale'
]

DEFAULT_THRESHOLD = 0.6  # From Kraus et al. (2025): optimal balance of quality vs. LLM cost


# ── Difference categorization ──────────────────────────────────────────────────

def categorize_difference(str1: str, str2: str, directionality: str = 'ltr') -> str:
	"""
	Categorize why two strings differ.

	Parameters
	----------
	str1, str2 : str
		The strings to compare
	directionality : str
		'ltr' (Latin, Cyrillic, etc. - may have case)
		'rtl' (Arabic, Hebrew, etc. - typically no case)

	Returns one of:
	  - 'capitalization': same text, different case (only checked for LTR scripts)
	  - 'whitespace': same text ignoring whitespace
	  - 'both': differs in both capitalization and whitespace (only for LTR)
	  - 'content': actual content difference
	"""
	if str1 == str2:
		return 'identical'

	# Only check capitalization for LTR scripts (which typically have case)
	# RTL scripts (Arabic, Hebrew, etc.) don't have capitalization
	if directionality == 'ltr':
		if str1.lower() == str2.lower():
			return 'capitalization'

	# Check whitespace only
	if str1.strip().lower() == str2.strip().lower():
		if str1.strip() == str2.strip():
			return 'whitespace'
		elif directionality == 'ltr':
			return 'both'  # both capitalization and whitespace
		else:
			return 'whitespace'  # RTL: only whitespace, no capitalization

	# Different content
	return 'content'


def analyze_differences(service_values: Dict[str, Optional[str]], directionality: str = 'ltr') -> Dict[str, str]:
	"""
	Analyze all pairwise differences in service outputs for a single row.

	Parameters
	----------
	service_values : dict
		{service_name: translation_string}
	directionality : str
		'ltr' or 'rtl' - used to determine if case-checking is relevant

	Returns dict of differences found.
	"""
	actual_values = {k: v for k, v in service_values.items() if v is not None}

	if len(actual_values) <= 1:
		return {'summary': 'no_differences'}  # Only one service or all empty

	differences = {}
	service_list = list(actual_values.items())

	for i, (svc1, val1) in enumerate(service_list):
		for svc2, val2 in service_list[i+1:]:
			if val1 != val2:
				diff_type = categorize_difference(val1, val2, directionality)
				key = f'{svc1}_vs_{svc2}'
				differences[key] = diff_type

	if not differences:
		differences['summary'] = 'all_identical'
	else:
		# Summarize types of differences
		diff_types = set(differences.values())
		differences['summary'] = ','.join(sorted(diff_types))

	return differences


# ── Core scoring logic ─────────────────────────────────────────────────────────

def score_row_confidence(
    row: pd.Series,
    service_cols: Dict[str, str],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Compute frequency-based confidence for a single translation row.

    Parameters
    ----------
    row : pd.Series
    service_cols : dict
        Mapping of {service_name: column_name} to include in scoring.
    threshold : float

    Returns
    -------
    dict with keys:
        best_candidate, confidence, total_services, agreeing_services,
        candidate_distribution, above_threshold, unique_candidates
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
            'above_threshold': False,
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
        'above_threshold': confidence >= threshold,
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
    """Load the initial_translated_terms CSV for a term/variant combination."""
    if variant == 'comparative':
        path = os.path.join(
            data_directory_path, 'metadata_files', 'translated_terms',
            term_slug, 'initial_translated_terms.csv'
        )
    else:
        path = os.path.join(
            data_directory_path, 'metadata_files', 'translated_terms',
            term_slug, 'prompt_variants', f'{variant}_initial_translated_terms.csv'
        )

    if not os.path.exists(path):
        return None

    try:
        df = read_csv_file(path)
        df['prompt_variant'] = variant
        return df
    except Exception as e:
        console.print(f"⚠ Could not load {path}: {e}", style="bold yellow")
        return None


def score_dataframe(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Add confidence scoring columns to a translations DataFrame.

    Computes both primary-only confidence (GT + EasyNMT + Wikipedia) and
    full confidence including LLM outputs, so you can compare.
    """
    # Primary service confidence
    primary_scores = df.apply(
        lambda row: score_row_confidence(row, PRIMARY_SERVICES, threshold),
        axis=1
    )
    primary_df = pd.DataFrame(primary_scores.tolist())
    primary_df.columns = [f'primary_{c}' for c in primary_df.columns]

    # All-service confidence (primary + LLM)
    all_services = {**PRIMARY_SERVICES, **LLM_SERVICES}
    # Filter to only columns present in df
    available_services = {k: v for k, v in all_services.items() if v in df.columns}
    all_scores = df.apply(
        lambda row: score_row_confidence(row, available_services, threshold),
        axis=1
    )
    all_df = pd.DataFrame(all_scores.tolist())
    all_df.columns = [f'all_{c}' for c in all_df.columns]

    result = pd.concat([df.reset_index(drop=True), primary_df, all_df], axis=1)
    return result


def compute_summary(scored_df: pd.DataFrame, variant: str, term: str) -> dict:
    """Compute aggregate stats for a single variant/term combination."""
    total = len(scored_df)
    has_any = scored_df['primary_total_services'] > 0

    return {
        'term': term,
        'variant': variant,
        'total_languages': total,
        'languages_with_primary': int(has_any.sum()),
        'high_confidence_primary': int(scored_df['primary_above_threshold'].sum()),
        'low_confidence_primary': int((has_any & ~scored_df['primary_above_threshold']).sum()),
        'no_primary_data': int((~has_any).sum()),
        'mean_confidence': round(scored_df.loc[has_any, 'primary_confidence'].mean(), 4) if has_any.any() else None,
        'median_confidence': round(scored_df.loc[has_any, 'primary_confidence'].median(), 4) if has_any.any() else None,
        'pct_above_threshold': round(scored_df['primary_above_threshold'].sum() / total * 100, 1),
        'high_llm_disagreement': int(
            (scored_df['primary_unique_candidates'] > 1).sum()
        ),
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_confidence_evaluation(
    data_directory_path: str,
    target_terms: List[str],
    variants: List[str] = ALL_VARIANTS,
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run confidence scoring for all term/variant combinations.

    Returns
    -------
    scored_df : pd.DataFrame
        All rows with confidence scores appended.
    summary_df : pd.DataFrame
        One row per term/variant with aggregate stats.
    low_confidence_df : pd.DataFrame
        Rows below threshold — potential hallucinations or absent concepts.
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
            scored = score_dataframe(df, threshold=threshold)
            all_scored.append(scored)
            summaries.append(compute_summary(scored, variant, term))

    if not all_scored:
        console.print("⚠ No data found for any term/variant combination.", style="bold red")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    scored_df = pd.concat(all_scored, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    # Low-confidence rows: primary services produced data but disagreed
    low_confidence_df = scored_df[
        (scored_df['primary_total_services'] > 0) &
        (~scored_df['primary_above_threshold'])
    ].copy()

    # Save outputs
    _dir = output_dir or os.path.join(
        data_directory_path, 'metadata_files', 'evaluation'
    )
    os.makedirs(_dir, exist_ok=True)

    scored_path   = os.path.join(_dir, 'confidence_scores.csv')
    summary_path  = os.path.join(_dir, 'confidence_summary.csv')
    low_conf_path = os.path.join(_dir, 'low_confidence.csv')

    scored_df.to_csv(scored_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    low_confidence_df.to_csv(low_conf_path, index=False)

    console.print(f"\n✓ Outputs written to: {_dir}", style="bold green")
    console.print(f"  confidence_scores.csv  : {len(scored_df)} rows", style="green")
    console.print(f"  confidence_summary.csv : {len(summary_df)} rows", style="green")
    console.print(f"  low_confidence.csv     : {len(low_confidence_df)} rows", style="green")

    # Print summary table to console
    _print_summary_table(summary_df)

    return scored_df, summary_df, low_confidence_df


def _print_summary_table(summary_df: pd.DataFrame) -> None:
    table = Table(title="Confidence Scoring Summary", show_header=True)
    for col in ['term', 'variant', 'total_languages', 'high_confidence_primary',
                'low_confidence_primary', 'pct_above_threshold', 'mean_confidence']:
        table.add_column(col, style="cyan" if col in ('term', 'variant') else "white")
    for _, row in summary_df.iterrows():
        table.add_row(*[str(row.get(c, '')) for c in [
            'term', 'variant', 'total_languages', 'high_confidence_primary',
            'low_confidence_primary', 'pct_above_threshold', 'mean_confidence'
        ]])
    console.print(table)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score translation agreement across primary services."
    )
    parser.add_argument(
        '--term', nargs='+',
        default=['Computational Humanities', 'Digital Humanities'],
        help='Target terms to evaluate'
    )
    parser.add_argument(
        '--variants', nargs='+', default=ALL_VARIANTS,
        choices=ALL_VARIANTS,
        help='Prompt variants to include'
    )
    parser.add_argument(
        '--threshold', type=float, default=DEFAULT_THRESHOLD,
        help='Confidence threshold (default: 0.6, per Kraus et al. 2025)'
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory (default: data_dir/metadata_files/evaluation/)'
    )
    args = parser.parse_args()

    data_dir = get_data_directory_path()
    run_confidence_evaluation(
        data_directory_path=data_dir,
        target_terms=args.term,
        variants=args.variants,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()