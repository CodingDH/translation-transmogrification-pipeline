#!/usr/bin/env python3
"""
explore_confidence_across_variants.py
=======================================
Measure translation consistency ACROSS prompt variants.

For each (language × service) combination, compare the translation produced by
each prompt variant. This measures prompt robustness — does the same LLM give
the same answer regardless of how it was prompted?

  - Is Claude's German translation stable across minimal vs github_searcher?
  - Are local models (Llama, Gemma, Qwen, Mistral) more prompt-sensitive than
    API models (OpenAI, Claude, Gemini, DeepSeek)?
  - Which services are most sensitive to prompt wording?
  - Which language families show the most prompt-driven variation?

LLM services compared: OpenAI, Claude, Gemini, DeepSeek (API) and
Llama, Gemma, Qwen, Mistral (local Ollama).

Non-LLM sources — MT baselines (GT, EasyNMT, Lingvanex) and the Wikipedia
community reference — are prompt-invariant by design and should always score
1.0 — they serve as a sanity check.

Compare with: explore_confidence_within_variant.py (which measures if different
services agree *within* a single prompt variant).

Output files are written to data_directory/translated_terms/{term}/evaluation/:
  - `across_variant_detail.csv`: one row per language × service
  - `across_variant_service_summary.csv`: aggregate stats per service

Usage:
    python explore_confidence_across_variants.py
    python explore_confidence_across_variants.py --term "Computational Humanities"
    python explore_confidence_across_variants.py --output-dir /path/to/out
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.utils import get_data_directory_path, get_language_family, analyze_differences
from scripts.exploration.explore_confidence_within_variant import load_variant_df

console = Console()

ALL_VARIANTS = [
    'minimal', 'fluent_speaker', 'github_searcher', 'judge'
]

# Services whose translations vary by prompt variant (LLMs)
LLM_SERVICES: Dict[str, str] = {
    'OpenAI':   'openai_translated_term',
    'Claude':   'claude_translated_term',
    'Gemini':   'gemini_translated_term',
    'DeepSeek': 'deepseek_translated_term',
    'Llama':    'llama_translated_term',
    'Gemma':    'gemma_translated_term',
    'Qwen':     'qwen_translated_term',
    'Mistral':  'mistral_translated_term',
}

# Services that are prompt-invariant — should always score 1.0 (sanity check)
BASELINE_SERVICES: Dict[str, str] = {
    'Wikipedia':      'wikipedia_translated_term',
    'Google Translate': 'gt_translated_term',
    'EasyNMT':        'enmt_translated_term',
    'Lingvanex':      'lingvanex_translated_term',
}

ALL_SERVICES = {**BASELINE_SERVICES, **LLM_SERVICES}


# ── Core logic ──────────────────────────────────────────────────────────────

def score_service_across_variants(translations: Dict[str, Optional[str]]) -> Dict:
    """
    For a single (language × service), score agreement across variants.

    Parameters
    ----------
    translations : dict
        {variant_name: translated_term_or_None}

    Returns
    -------
    dict with agreement_rate, best_candidate, n_variants_present, n_agree, etc.
    """
    valid = {v: t for v, t in translations.items()
             if t and isinstance(t, str) and t.strip() and t != 'nan'}

    if not valid:
        return {
            'n_variants_present': 0,
            'n_agree': 0,
            'agreement_rate': None,
            'best_candidate': None,
            'variant_translations': str(translations),
        }

    counts: Dict[str, int] = defaultdict(int)
    for t in valid.values():
        counts[t] += 1

    best = max(counts, key=counts.__getitem__)
    n_agree = counts[best]
    rate = round(n_agree / len(valid), 4)

    differences = analyze_differences(valid)

    return {
        'n_variants_present': len(valid),
        'n_agree': n_agree,
        'agreement_rate': rate,
        'best_candidate': best,
        'variant_translations': str({v: t for v, t in valid.items()}),
        'difference_types': differences.get('summary', 'unknown'),
    }


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_across_variant_evaluation(
    data_directory_path: str,
    target_terms: List[str],
    variants: List[str] = ALL_VARIANTS,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (language × service), compare translations across prompt variants.
    
    Parameters
    ----------
    data_directory_path : str
		Root path to the dataset directory.
    target_terms : list of str
		List of target terms to evaluate (e.g., ['Digital Humanities']).
    variants : list of str
		Which prompt variants to include in the comparison.
    output_dir : str, optional
		Directory to write output CSVs. Defaults to a subdirectory of data_directory_path.

    Returns
    -------
    detail_df : pd.DataFrame
        One row per (language × service). Columns include language_code,
        language_name, language_family, service, n_variants_present,
        n_agree, agreement_rate, best_candidate.
    service_summary_df : pd.DataFrame
        One row per service with mean, std, and CV of agreement rates.
    """
    all_rows = []

    for term in target_terms:
        term_slug = term.lower().replace(' ', '_')
        console.print(f"\n📊 Cross-variant scoring: {term}", style="bold cyan")

        variant_dfs: Dict[str, pd.DataFrame] = {}
        for variant in variants:
            df = load_variant_df(data_directory_path, term_slug, variant)
            if df is not None:
                variant_dfs[variant] = df
                console.print(f"  ✓ Loaded {len(df)} rows for '{variant}'", style="green")
            else:
                console.print(f"  ⚠ No data for '{variant}' — skipping", style="dim yellow")

        if not variant_dfs:
            console.print(f"  ⚠ No data found for '{term}'", style="bold yellow")
            continue

        # Build a union of all language_codes × term_sources
        ref_df = next(iter(variant_dfs.values()))
        lang_keys = ref_df[['language_code', 'language_name', 'term_source']].drop_duplicates()

        for _, lang_row in lang_keys.iterrows():
            lang_code = str(lang_row['language_code'])
            lang_name = str(lang_row.get('language_name', lang_code))
            term_source = str(lang_row['term_source'])
            lang_family = get_language_family(lang_code)

            for service_name, col in ALL_SERVICES.items():
                # Collect this service's translation from each variant
                translations: Dict[str, Optional[str]] = {}
                for variant, vdf in variant_dfs.items():
                    match = vdf[
                        (vdf['language_code'] == lang_code) &
                        (vdf['term_source'] == term_source)
                    ]
                    val = None
                    if not match.empty and col in match.columns:
                        raw = match.iloc[0][col]
                        val = str(raw) if pd.notna(raw) else None
                    translations[variant] = val

                result = score_service_across_variants(translations)
                result.update({
                    'term_source': term_source,
                    'language_code': lang_code,
                    'language_name': lang_name,
                    'language_family': lang_family,
                    'service': service_name,
                    'is_baseline': service_name in BASELINE_SERVICES,
                })
                all_rows.append(result)

    if not all_rows:
        console.print("⚠ No data found for any term/variant combination.", style="bold red")
        return pd.DataFrame(), pd.DataFrame()

    detail_df = pd.DataFrame(all_rows)

    # Service summary: only count rows where the service actually produced data
    has_data = detail_df[detail_df['n_variants_present'] > 0]
    service_summary = (
        has_data.groupby('service')
        .agg(
            n_languages=('language_code', 'nunique'),
            mean_agreement=('agreement_rate', 'mean'),
            std_agreement=('agreement_rate', 'std'),
            is_baseline=('is_baseline', 'first'),
        )
        .reset_index()
        .assign(
            mean_agreement=lambda d: d['mean_agreement'].round(4),
            std_agreement=lambda d: d['std_agreement'].round(4),
        )
        .sort_values('mean_agreement', ascending=False)
    )
    service_summary['cv_agreement'] = (
        service_summary['std_agreement'] / service_summary['mean_agreement']
    ).round(4)

    _dir = output_dir or os.path.join(data_directory_path, 'translated_terms', target_terms[0].lower().replace(' ', '_'), 'evaluation')
    os.makedirs(_dir, exist_ok=True)

    detail_df.to_csv(os.path.join(_dir, 'across_variant_detail.csv'), index=False)
    service_summary.to_csv(os.path.join(_dir, 'across_variant_service_summary.csv'), index=False)

    console.print(f"\n✓ Outputs written to: {_dir}", style="bold green")
    console.print(f"  across_variant_detail.csv          : {len(detail_df)} rows", style="green")
    console.print(f"  across_variant_service_summary.csv : {len(service_summary)} rows", style="green")

    _print_summary_table(service_summary)

    return detail_df, service_summary


def _print_summary_table(summary_df: pd.DataFrame) -> None:
    table = Table(title="Cross-Variant Service Stability", show_header=True)
    cols = ['service', 'n_languages', 'mean_agreement', 'std_agreement', 'cv_agreement']
    for col in cols:
        table.add_column(col, style="cyan" if col == 'service' else "white")
    for _, row in summary_df.iterrows():
        table.add_row(*[str(row.get(c, '')) for c in cols])
    console.print(table)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare translation consistency across prompt variants per service."
    )
    parser.add_argument('--term', nargs='+', default=['Digital Humanities'])
    parser.add_argument('--variants', nargs='+', default=ALL_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    data_dir = get_data_directory_path()
    run_across_variant_evaluation(
        data_directory_path=data_dir,
        target_terms=args.term,
        variants=args.variants,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
