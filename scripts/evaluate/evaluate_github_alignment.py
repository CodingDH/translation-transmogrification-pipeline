#!/usr/bin/env python3
"""
evaluate_github_alignment.py
=============================
Compare pipeline translation outputs to actual community usage as evidenced
by GitHub repository metadata (descriptions, README text, topics).

This is your methodological contribution that no existing translation paper
makes. WOKIE evaluates quality by back-translating against known reference
labels. This script evaluates quality by asking: does the translation match
how the scholarly community in this language actually names the concept in
their practice?

Three alignment signals, each capturing something different:
  1. EXACT MATCH    — pipeline output appears literally in GitHub metadata
  2. FUZZY MATCH    — Levenshtein similarity ≥ 0.8 (close variants)
  3. LOAN WORD      — the English source term appears, signalling the community
                      uses the borrowed term rather than a native equivalent
                      (this is not a failure — it is a finding)

Cases where all pipeline variants produce translations with no GitHub signal
at all are candidates for "absent concept" analysis — the pipeline may have
confabulated a plausible-sounding translation for a concept that hasn't
been adopted by that community.

Requires:
  - Pipeline output CSVs from generate_translations_prompts.py
  - GitHub repo/user data from the CodingDH data collection pipeline
    (the repo data collected by utils.py and friends in the parent project)

Output (data_dir/metadata_files/evaluation/):
  - github_alignment.csv        : per-row alignment scores
  - github_alignment_summary.csv: aggregate stats by language family + variant
  - absent_concept_candidates.csv: rows with no GitHub signal across all variants
  - loan_word_cases.csv         : communities using the English term directly

Usage:
    python evaluate_github_alignment.py
    python evaluate_github_alignment.py --term "Digital Humanities"
    python evaluate_github_alignment.py --github-field description topics
"""

import argparse
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rich.console import Console
from rich.progress import track

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generation_scripts.utils import get_data_directory_path, read_csv_file, read_combine_files

console = Console()

ALL_VARIANTS = ['comparative', 'minimal', 'expert_persona', 'contextual', 'native_rationale']

# GitHub metadata fields to search in. 'topics' is the most deliberate signal
# (users explicitly tag repos); 'description' is natural language usage.
GITHUB_FIELDS = ['description', 'topics', 'name']

# Language family groupings for aggregate analysis
# (iso 639-1 codes → family label)
LANGUAGE_FAMILIES = {
    'Romance':    ['es', 'fr', 'it', 'pt', 'ro', 'ca', 'gl', 'oc'],
    'Germanic':   ['de', 'en', 'nl', 'sv', 'da', 'no', 'fi', 'af'],
    'Slavic':     ['ru', 'pl', 'cs', 'sk', 'bg', 'hr', 'sr', 'uk', 'sl'],
    'East Asian': ['zh', 'ja', 'ko'],
    'Semitic':    ['ar', 'he'],
    'South Asian':['hi', 'bn', 'ur', 'ta', 'te', 'ml'],
    'Other':      [],  # catch-all
}


def _family_for_code(language_code: str) -> str:
    for family, codes in LANGUAGE_FAMILIES.items():
        if language_code in codes:
            return family
    return 'Other'


def _normalise(text: str) -> str:
    """Lowercase, strip accents, normalise whitespace."""
    text = text.lower().strip()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'\s+', ' ', text)
    return text


def _levenshtein(a: str, b: str) -> float:
    """Normalised Levenshtein similarity (0=totally different, 1=identical)."""
    if not a or not b:
        return 0.0
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j-1], prev[j], dp[j-1])
    distance = dp[n]
    return 1.0 - (distance / max(m, n))


def load_github_data(
    data_directory_path: str,
    language_code: str,
    fields: List[str] = GITHUB_FIELDS,
) -> Set[str]:
    """
    Load GitHub repo metadata for a given language code and return a set
    of normalised text tokens found in the specified fields.

    Looks for repo CSV files collected by the CodingDH data pipeline.
    These are stored in:
      data_dir/historic_data/entity_files/all_repos/collected_datasets/
    """
    repos_dir = os.path.join(
        data_directory_path, 'historic_data', 'entity_files',
        'all_repos', 'collected_datasets'
    )
    if not os.path.exists(repos_dir):
        return set()

    # Filter repo files by language code — CodingDH stores repos per org/user
    # so we load all and filter by the 'language' field
    try:
        all_files = [f for f in os.listdir(repos_dir) if f.endswith('.csv')]
        if not all_files:
            return set()

        # Load a sample to check structure
        sample_path = os.path.join(repos_dir, all_files[0])
        sample = read_csv_file(sample_path)

        tokens: Set[str] = set()
        for fname in all_files:
            try:
                df = read_csv_file(os.path.join(repos_dir, fname))
                # Filter to repos in the target language if language column exists
                if 'language' in df.columns:
                    # GitHub 'language' is programming language not natural language
                    # We use it as a weak proxy — it's not ideal
                    pass

                for field in fields:
                    if field in df.columns:
                        vals = df[field].dropna().astype(str)
                        for val in vals:
                            tokens.add(_normalise(val))
            except Exception:
                continue
        return tokens

    except Exception as e:
        console.print(f"  ⚠ Could not load GitHub data: {e}", style="dim yellow")
        return set()


def load_github_text_by_language(
    data_directory_path: str,
    language_code: str,
    fields: List[str] = GITHUB_FIELDS,
) -> List[str]:
    """
    Load GitHub metadata for repos associated with a given natural language.

    The CodingDH pipeline collects repos from users/orgs filtered by
    natural language (stored in search query CSVs). This function looks
    for that structured data.

    Falls back to searching all repos if language-specific data isn't available.
    """
    # Look for language-specific search results
    search_path = os.path.join(
        data_directory_path, 'metadata_files',
        'search_terms_results', f'{language_code}_repos.csv'
    )
    if os.path.exists(search_path):
        try:
            df = read_csv_file(search_path)
            texts = []
            for field in fields:
                if field in df.columns:
                    texts.extend(df[field].dropna().astype(str).tolist())
            return texts
        except Exception:
            pass

    # Fallback: look in the combined processed datasets
    processed_path = os.path.join(
        data_directory_path, 'metadata_files', 'translated_terms',
        'computational_humanities', 'github_repos.csv'
    )
    if os.path.exists(processed_path):
        try:
            df = read_csv_file(processed_path)
            if 'language_code' in df.columns:
                df = df[df['language_code'] == language_code]
            texts = []
            for field in fields:
                if field in df.columns:
                    texts.extend(df[field].dropna().astype(str).tolist())
            return texts
        except Exception:
            pass

    return []


def check_alignment(
    translation: Optional[str],
    source_term: str,
    github_texts: List[str],
    fuzzy_threshold: float = 0.8,
) -> dict:
    """
    Check how well a translation aligns with GitHub community usage.

    Returns
    -------
    dict with:
        exact_match: bool — translation appears literally in GitHub data
        fuzzy_match: bool — translation appears with similarity ≥ threshold
        max_similarity: float — best fuzzy similarity score found
        loan_word: bool — English source term appears (community uses borrowed term)
        github_evidence: str — the matching GitHub text snippet, if any
        signal_strength: 'strong'|'weak'|'loan'|'absent'
    """
    if not translation or not isinstance(translation, str):
        return {
            'exact_match': False, 'fuzzy_match': False,
            'max_similarity': 0.0, 'loan_word': False,
            'github_evidence': '', 'signal_strength': 'no_translation',
        }

    norm_translation = _normalise(translation)
    norm_source = _normalise(source_term)
    exact = False
    max_sim = 0.0
    evidence = ''
    loan = False

    for text in github_texts:
        norm_text = _normalise(text)

        # Exact match
        if norm_translation in norm_text:
            exact = True
            # Extract snippet
            idx = norm_text.find(norm_translation)
            start = max(0, idx - 30)
            end = min(len(text), idx + len(translation) + 30)
            evidence = f"...{text[start:end]}..."
            break

        # Loan word check — English term appears in this language's GitHub data
        if norm_source in norm_text:
            loan = True

        # Fuzzy match
        words = re.findall(r'\w+', norm_text)
        for window_size in range(1, min(4, len(words) + 1)):
            for i in range(len(words) - window_size + 1):
                phrase = ' '.join(words[i:i + window_size])
                sim = _levenshtein(norm_translation, phrase)
                if sim > max_sim:
                    max_sim = sim
                    if sim >= fuzzy_threshold:
                        evidence = phrase

    fuzzy = max_sim >= fuzzy_threshold and not exact

    if exact:
        signal = 'strong'
    elif fuzzy:
        signal = 'weak'
    elif loan:
        signal = 'loan'
    else:
        signal = 'absent'

    return {
        'exact_match': exact,
        'fuzzy_match': fuzzy,
        'max_similarity': round(max_sim, 4),
        'loan_word': loan,
        'github_evidence': evidence[:200] if evidence else '',
        'signal_strength': signal,
    }


def load_variant_translations(
    data_directory_path: str,
    term_slug: str,
    variants: List[str] = ALL_VARIANTS,
) -> pd.DataFrame:
    """
    Load all variant translation CSVs and combine into a wide DataFrame.
    One row per language, columns for each variant's primary translation term.
    """
    dfs = {}
    for variant in variants:
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
        if os.path.exists(path):
            df = read_csv_file(path)
            # Use the consensus 'term' column as the variant's translation
            if 'term' in df.columns:
                df = df[['language_code', 'language_name', 'term_source', 'term']].copy()
                df = df.rename(columns={'term': f'{variant}_term'})
                dfs[variant] = df

    if not dfs:
        return pd.DataFrame()

    merged = None
    base_cols = ['language_code', 'language_name', 'term_source']
    for variant, df in dfs.items():
        if merged is None:
            merged = df
        else:
            on = [c for c in base_cols if c in merged.columns and c in df.columns]
            merged = merged.merge(
                df[on + [f'{variant}_term']], on=on, how='outer'
            )
    return merged


def run_github_alignment(
    data_directory_path: str,
    target_terms: List[str],
    variants: List[str] = ALL_VARIANTS,
    github_fields: List[str] = GITHUB_FIELDS,
    fuzzy_threshold: float = 0.8,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run GitHub alignment evaluation for all terms and languages.

    Returns
    -------
    alignment_df : pd.DataFrame
    summary_df : pd.DataFrame
    absent_df : pd.DataFrame
    loan_word_df : pd.DataFrame
    """
    _dir = output_dir or os.path.join(data_directory_path, 'metadata_files', 'evaluation')
    os.makedirs(_dir, exist_ok=True)

    all_rows = []

    for term in target_terms:
        term_slug = term.lower().replace(' ', '_')
        console.print(f"\n🔗 GitHub alignment for: {term}", style="bold cyan")

        translations_df = load_variant_translations(data_directory_path, term_slug, variants)
        if translations_df.empty:
            console.print(f"  ⚠ No translation data found for '{term}'", style="bold yellow")
            continue

        console.print(f"  {len(translations_df)} languages to evaluate", style="green")

        for _, row in track(
            translations_df.iterrows(),
            total=len(translations_df),
            description="Checking GitHub alignment...",
        ):
            lang_code = str(row.get('language_code', ''))
            lang_name = str(row.get('language_name', ''))

            # Load GitHub texts for this language
            github_texts = load_github_text_by_language(
                data_directory_path, lang_code, github_fields
            )

            result_row = {
                'term_source': term,
                'language_code': lang_code,
                'language_name': lang_name,
                'language_family': _family_for_code(lang_code),
                'github_text_count': len(github_texts),
            }

            for variant in variants:
                col = f'{variant}_term'
                translation = row.get(col)
                alignment = check_alignment(
                    translation, term, github_texts, fuzzy_threshold
                )
                result_row[f'{variant}_translation'] = translation
                result_row[f'{variant}_exact'] = alignment['exact_match']
                result_row[f'{variant}_fuzzy'] = alignment['fuzzy_match']
                result_row[f'{variant}_similarity'] = alignment['max_similarity']
                result_row[f'{variant}_signal'] = alignment['signal_strength']
                result_row[f'{variant}_evidence'] = alignment['github_evidence']
                result_row[f'{variant}_loan_word'] = alignment['loan_word']

            # Aggregate signals
            any_signal = any(
                result_row.get(f'{v}_signal') in ('strong', 'weak')
                for v in variants if f'{v}_signal' in result_row
            )
            any_loan = any(
                result_row.get(f'{v}_loan_word', False) for v in variants
            )
            result_row['any_github_signal'] = any_signal
            result_row['loan_word_community'] = any_loan
            result_row['absent_concept_candidate'] = (
                not any_signal and not any_loan and len(github_texts) > 0
            )

            all_rows.append(result_row)

    if not all_rows:
        console.print("⚠ No alignment results produced.", style="bold red")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    alignment_df = pd.DataFrame(all_rows)

    # Subset DataFrames
    absent_df = alignment_df[alignment_df['absent_concept_candidate'] == True].copy()
    loan_df = alignment_df[alignment_df['loan_word_community'] == True].copy()

    # Summary by language family and variant
    summary_rows = []
    for family in alignment_df['language_family'].unique():
        fam_df = alignment_df[alignment_df['language_family'] == family]
        for variant in variants:
            if f'{variant}_signal' not in fam_df.columns:
                continue
            summary_rows.append({
                'term': alignment_df['term_source'].iloc[0] if len(alignment_df) else '',
                'language_family': family,
                'variant': variant,
                'n_languages': len(fam_df),
                'n_with_github_data': int((fam_df['github_text_count'] > 0).sum()),
                'strong_match': int((fam_df[f'{variant}_signal'] == 'strong').sum()),
                'weak_match': int((fam_df[f'{variant}_signal'] == 'weak').sum()),
                'loan_word': int((fam_df[f'{variant}_signal'] == 'loan').sum()),
                'absent': int((fam_df[f'{variant}_signal'] == 'absent').sum()),
                'mean_similarity': round(fam_df[f'{variant}_similarity'].mean(), 4),
            })
    summary_df = pd.DataFrame(summary_rows)

    # Save outputs
    alignment_df.to_csv(os.path.join(_dir, 'github_alignment.csv'), index=False)
    summary_df.to_csv(os.path.join(_dir, 'github_alignment_summary.csv'), index=False)
    absent_df.to_csv(os.path.join(_dir, 'absent_concept_candidates.csv'), index=False)
    loan_df.to_csv(os.path.join(_dir, 'loan_word_cases.csv'), index=False)

    console.print(f"\n✓ Alignment results saved to: {_dir}", style="bold green")
    console.print(f"  github_alignment.csv        : {len(alignment_df)} rows", style="green")
    console.print(f"  absent_concept_candidates   : {len(absent_df)} rows", style="green")
    console.print(f"  loan_word_cases             : {len(loan_df)} rows", style="green")

    # Quick summary to console
    if len(alignment_df):
        total = len(alignment_df)
        console.print(f"\n  Of {total} languages:", style="cyan")
        console.print(f"    {len(absent_df)} absent concept candidates ({round(len(absent_df)/total*100,1)}%)", style="cyan")
        console.print(f"    {len(loan_df)} loan word communities ({round(len(loan_df)/total*100,1)}%)", style="cyan")

    return alignment_df, summary_df, absent_df, loan_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare pipeline translations to GitHub community usage."
    )
    parser.add_argument('--term', nargs='+',
                        default=['Computational Humanities', 'Digital Humanities'])
    parser.add_argument('--variants', nargs='+', default=ALL_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument('--github-field', nargs='+', default=GITHUB_FIELDS,
                        choices=['description', 'topics', 'name', 'readme'])
    parser.add_argument('--fuzzy-threshold', type=float, default=0.8)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    data_dir = get_data_directory_path()
    run_github_alignment(
        data_directory_path=data_dir,
        target_terms=args.term,
        variants=args.variants,
        github_fields=args.github_field,
        fuzzy_threshold=args.fuzzy_threshold,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()