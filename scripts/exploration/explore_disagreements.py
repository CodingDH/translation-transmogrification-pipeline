#!/usr/bin/env python3
"""
analyze_disagreements.py
========================
Classify translation disagreements into four uncertainty categories:

  MEASUREMENT_ARTEFACT — apparent disagreement that dissolves after normalization (e.g. Esperanto Ciferecaj vs Cifereca — one letter of grammatical agreement). The scoring tool's limit, not a real divergence.

  STRUCTURAL_ABSENCE — the concept has no community equivalent; model signals this through loan words, explicit "no equivalent" rationale text, or near-zero coverage. Swahili: EasyNMT returns "electronic natural resources"; OpenAI keeps "Humanities" in English.

  PRODUCTIVE_DISAGREEMENT — multiple legitimate in-language alternatives reflecting a real scholarly debate.  Wikipedia presence is the strongest positive signal. Arabic: إنسانيات vs العلوم الإنسانية — both valid, both used, community has not converged.

  TRANSMOGRIFICATION — confident, fluent, elaborate output untethered from any community practice; rationales become the tell.  Gothic: five different outputs, all internally reasoned, none verifiable.

NOT_APPLICABLE is used when all services agree (no disagreement to classify).

Classification hierarchy (applied in order, first match wins):
  1. All translations are absent/identical → NOT_APPLICABLE
  2. Disagreement collapses after normalization → MEASUREMENT_ARTEFACT
  3. Loan-word or rationale-based absence signals → STRUCTURAL_ABSENCE
  4. Wikipedia present + multiple in-language outputs → PRODUCTIVE_DISAGREEMENT
  5. Default → TRANSMOGRIFICATION

Input
-----
  - translated_terms/{term}/evaluation/across_variant_detail.csv
  - translated_terms/{term}/{direct,prompt}_services/*.csv (for rationales)
    loaded via load_variant_df for a single reference variant

Output
------
  translated_terms/{term}/evaluation/disagreement_analysis.csv
    One row per (language_code × term_source) with category, evidence columns,
    and extracted rationale snippets.

Usage
-----
    python analyze_disagreements.py
    python analyze_disagreements.py --term "Digital Humanities"
    python analyze_disagreements.py --term "Digital Humanities" --variant comparative
    python analyze_disagreements.py --threshold 0.15
"""

import argparse
import os
import re
import sys
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.utils import (
    get_data_directory_path, read_csv_file,
    detect_script_disagreement,
)
from scripts.exploration.explore_confidence_within_variant import load_variant_df

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────

CATEGORIES = {
    'NOT_APPLICABLE':        'All services agree — no disagreement to classify',
    'MEASUREMENT_ARTEFACT':  'Disagreement dissolves after normalization',
    'STRUCTURAL_ABSENCE':    'Concept absent from community; model borrows or signals absence',
    'PRODUCTIVE_DISAGREEMENT': 'Multiple legitimate in-language alternatives',
    'TRANSMOGRIFICATION':    'Confident fluent output untethered from community practice',
}

# Keyword matching assumes English-language rationales. Use 'comparative' or
# any variant other than 'native_rationale' as the rationale source — those
# variants ask models to reason in English regardless of target language.
# 'native_rationale' produces rationales in the target language; keywords will
# silently miss signals for non-Latin-script languages if that variant is used.
RATIONALE_COLS = {
    'Claude':  'claude_translation_rationale',
    'OpenAI':  'openai_translation_rationale',
    'Gemini':  'gemini_translation_rationale',
    'Ollama':  'ollama_translation_rationale',
}

TRANSLATION_COLS = {
    'Wikipedia':        'wikipedia_translated_term',
    'Google Translate': 'gt_translated_term',
    'EasyNMT':          'enmt_translated_term',
    'Lingvanex':        'lingvanex_translated_term',
    'OpenAI':           'openai_translated_term',
    'Claude':           'claude_translated_term',
    'Gemini':           'gemini_translated_term',
    'Ollama':           'ollama_translated_term',
}

# Keywords in rationale text that signal the model knows there is no equivalent
ABSENCE_KEYWORDS = [
    'no equivalent', 'no direct', 'no established', 'no widely', 'not adopted',
    'not institutionalized', 'not institutionalised', 'not yet adopted',
    'does not exist', "hasn't been", 'have not been', 'has not been',
    'no community', 'no scholarly', 'not used in', 'concept does not',
    'term does not', 'no translation', 'untranslated', 'no native',
    'no recognized', 'no recognised', 'borrowed', 'loanword', 'loan word',
    'borrowing', 'anglicism', 'anglicization', 'anglicised',
    'transliterat', 'keep.*english', 'retain.*english',
]

# Keywords suggesting elaborate construction — transmogrification signal
CONSTRUCTION_KEYWORDS = [
    'etymolog', 'root word', 'compound', 'combining', 'derive from',
    'ancient', 'classical', 'proto-', 'reconstructed', 'neologism',
    'coined', 'coining', 'create.*term', 'construct', 'formation',
    'morpholog', 'word-forming', 'suffix', 'prefix',
]

# Keywords suggesting community debate — productive disagreement signal
DEBATE_KEYWORDS = [
    'some scholars', 'different communities', 'preferred by', 'others use',
    'alternatively', 'debate', 'contested', 'competing', 'variant',
    'both terms', 'either', 'also used', 'sometimes rendered',
]

# Source-language tokens that signal loan-word usage (case-insensitive)
SOURCE_TOKENS = ['digital', 'humanities', 'humanit', 'numérique', 'digitale', 'dh']


# ── String normalization ───────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip diacritics, remove punctuation, collapse whitespace.

    RTL scripts (Arabic, Hebrew) are handled correctly: Python's \\w is
    Unicode-aware so Arabic/Hebrew letters are preserved by the punctuation
    strip, and NFKD decomposition correctly normalises Arabic compatibility
    forms. The only combining characters removed are vowel-pointing marks
    (harakat, niqqud) which do not appear in formal scholarly translations.
    """
    if not text or not isinstance(text, str):
        return ''
    nfkd = unicodedata.normalize('NFKD', text.lower())
    no_combining = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', no_combining)).strip()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def max_pairwise_edit_distance(strings: List[str]) -> int:
    """Max edit distance among all pairs in a list of strings."""
    if len(strings) <= 1:
        return 0
    return max(edit_distance(a, b) for i, a in enumerate(strings) for b in strings[i + 1:])


def contains_source_token(
    text: str,
    source_tokens: List[str] = SOURCE_TOKENS,
) -> bool:
    """Return True if a full word in the translation matches a source-language token.

    Uses whole-word matching so that adapted forms like 'digitala' (Gothic
    inflection of 'digital') are not flagged — only unadapted borrowings like
    'digital' or 'humanities' appearing as standalone words.
    """
    if not text:
        return False
    words = set(normalize(text).split())
    return any(tok in words for tok in source_tokens)


# ── Rationale similarity ──────────────────────────────────────────────────

def compute_rationale_similarity(
    rationales: Dict[str, Optional[str]]
) -> Optional[float]:
    """Mean pairwise TF-IDF cosine similarity across service rationales.

    Uses character n-grams (3–5) so the metric is script-agnostic and works
    for Arabic, CJK, and other non-Latin rationales as well as Latin ones.
    Within-language comparisons (Claude vs OpenAI on the same target language)
    are meaningful regardless of language: high similarity means the models
    converged on the same reasoning; low similarity means divergent framings.

    Returns None when fewer than 2 non-empty rationales are available.
    """
    texts = [
        t for t in rationales.values()
        if t and isinstance(t, str) and t.strip() and t not in ('nan', 'None')
    ]
    if len(texts) < 2:
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
        tfidf = vec.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)
        n = len(texts)
        upper = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
        return round(float(np.mean(upper)), 4) if upper else None
    except Exception:
        return None


# ── Rationale keyword matching ─────────────────────────────────────────────

def match_keywords(rationale: str, keywords: List[str]) -> List[str]:
    """Return which keywords (as regex patterns) match in rationale text."""
    if not rationale or not isinstance(rationale, str):
        return []
    rationale_lower = rationale.lower()
    return [kw for kw in keywords if re.search(kw, rationale_lower)]


def extract_rationale_signals(
    rationales: Dict[str, Optional[str]]
) -> Dict[str, List[str]]:
    """
    Scan all service rationales for absence, construction, and debate keywords.

    Returns dict with keys 'absence', 'construction', 'debate', each a list
    of (service, keyword) tuples found.
    """
    signals: Dict[str, List] = {'absence': [], 'construction': [], 'debate': []}
    for service, text in rationales.items():
        if not text:
            continue
        for kw in match_keywords(text, ABSENCE_KEYWORDS):
            signals['absence'].append((service, kw))
        for kw in match_keywords(text, CONSTRUCTION_KEYWORDS):
            signals['construction'].append((service, kw))
        for kw in match_keywords(text, DEBATE_KEYWORDS):
            signals['debate'].append((service, kw))
    return signals


# ── Core classification ────────────────────────────────────────────────────

def classify_language(
    translations: Dict[str, Optional[str]],
    rationales: Dict[str, Optional[str]],
    has_wikipedia: bool,
    norm_threshold: float = 0.15,
    source_tokens: List[str] = SOURCE_TOKENS,
) -> Dict:
    """
    Classify a single (language × term) disagreement.

    Parameters
    ----------
    translations : dict  service_name → best translation string or None
    rationales   : dict  service_name → rationale text or None
    has_wikipedia : bool  Wikipedia has a translation for this language
    norm_threshold : float
        Max edit distance (as fraction of mean length) to call an artefact.

    Returns
    -------
    dict with 'category', 'evidence', 'loan_words_found', 'absence_signals',
              'construction_signals', 'debate_signals', 'n_unique_raw',
              'n_unique_normalized', 'max_edit_distance'
    """
    valid = {s: t for s, t in translations.items()
             if t and isinstance(t, str) and t.strip() and t not in ('nan', 'None')}

    # Script disagreement detection — run early so all branches can use it
    script_info = detect_script_disagreement(valid)

    if not valid:
        return {
            'category': 'STRUCTURAL_ABSENCE',
            'evidence': 'No service produced a translation',
            'loan_words_found': False,
            'absence_signals': [],
            'construction_signals': [],
            'debate_signals': [],
            'n_unique_raw': 0,
            'n_unique_normalized': 0,
            'max_edit_distance': None,
            'has_script_disagreement': False,
            'scripts_found': [],
            'script_per_service': [],
        }

    raw_vals = list(valid.values())
    unique_raw = set(v.strip().lower() for v in raw_vals)
    norm_vals = [normalize(v) for v in raw_vals]
    unique_norm = set(v for v in norm_vals if v)

    n_unique_raw = len(unique_raw)
    n_unique_norm = len(unique_norm)

    # Compute max edit distance among normalized unique values
    norm_list = list(unique_norm)
    max_dist = max_pairwise_edit_distance(norm_list)
    mean_len = sum(len(v) for v in norm_list) / len(norm_list) if norm_list else 1
    dist_threshold = max(2, int(norm_threshold * mean_len))

    # Loan word and rationale signals
    loan_detected = any(contains_source_token(t, source_tokens) for t in valid.values())
    signals = extract_rationale_signals(rationales)

    script_suffix = (
        f'; script disagreement: {"+".join(script_info["scripts_found"])}'
        if script_info['has_script_disagreement'] else ''
    )

    def _base(category: str, evidence: str) -> Dict:
        return {
            'category': category,
            'evidence': evidence + script_suffix,
            'loan_words_found': loan_detected,
            'absence_signals': signals['absence'],
            'construction_signals': signals['construction'],
            'debate_signals': signals['debate'],
            'n_unique_raw': n_unique_raw,
            'n_unique_normalized': n_unique_norm,
            'max_edit_distance': max_dist,
            'has_script_disagreement': script_info['has_script_disagreement'],
            'scripts_found': script_info['scripts_found'],
            'script_per_service': script_info['script_per_service'],
        }

    # ── Step 1: all agree (no disagreement) ──────────────────────────────
    if n_unique_raw == 1:
        return _base('NOT_APPLICABLE', 'All services produced the same translation')

    # ── Step 2: measurement artefact (disagreement dissolves after normalization,
    #           or is explained entirely by script variant) ─────────────────
    is_script_only = (
        script_info['has_script_disagreement']
        and n_unique_norm <= 1
    )
    if n_unique_norm <= 1 or max_dist <= dist_threshold or is_script_only:
        reason = (
            f'Normalized to {n_unique_norm} unique value(s); '
            f'max edit distance {max_dist} ≤ threshold {dist_threshold}'
        )
        if is_script_only:
            reason = (
                f'Script-only disagreement ({"+".join(script_info["scripts_found"])}); '
                f'same term in {n_unique_raw} scripts'
            )
        return _base('MEASUREMENT_ARTEFACT', reason)

    # ── Step 3: productive disagreement ──────────────────────────────────
    # Check Wikipedia BEFORE loan-word detection: a language like German or
    # Swedish that has Wikipedia coverage and uses "Digital" as a loan word
    # in some services is showing real community debate, not structural absence.
    if has_wikipedia and n_unique_raw >= 2:
        return _base(
            'PRODUCTIVE_DISAGREEMENT',
            f'Wikipedia translation present; '
            f'{n_unique_raw} distinct outputs; '
            f'debate signals: {len(signals["debate"])}',
        )

    # ── Step 4: structural absence ────────────────────────────────────────
    # Key distinction from transmogrification: the model actually borrowed the
    # source term OR produced very little data — not just said "no equivalent"
    # while inventing elaborate constructed terms.
    # Rationale absence keywords alone are not enough: Gothic's Claude says
    # "no Gothic word for digital" then constructs 'digitala manna-kunþi' —
    # that is transmogrification, not structural absence.
    absence_score = len(signals['absence']) + (3 if loan_detected else 0)
    genuine_absence = (
        loan_detected                 # model borrowed the source term unadapted
        or len(valid) <= 2            # almost no services produced anything
        or (n_unique_raw <= 2 and len(signals['absence']) >= 2)  # said "no equivalent" AND agreed
    )
    if absence_score >= 2 and genuine_absence:
        return _base(
            'STRUCTURAL_ABSENCE',
            f'Loan word detected: {loan_detected}; '
            f'absence signals: {len(signals["absence"])}; '
            f'services with data: {len(valid)}',
        )

    # ── Step 5: transmogrification (default) ─────────────────────────────
    return _base(
        'TRANSMOGRIFICATION',
        f'No Wikipedia; {n_unique_raw} distinct outputs; '
        f'absence signals: {len(signals["absence"])}; '
        f'construction signals: {len(signals["construction"])}',
    )


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_disagreement_analysis(
    data_directory_path: str,
    target_terms: List[str],
    rationale_variant: str = 'minimal',
    norm_threshold: float = 0.15,
    source_tokens: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    For every (language × term), classify the disagreement pattern.

    Parameters
    ----------
    data_directory_path : str
    target_terms : List[str]
    rationale_variant : str
        Which prompt variant to use for extracting rationale text.
        Must be an English-reasoning variant — keyword matching is English-only.
        'minimal' is the default. Do NOT use 'native_rationale': that variant
        produces rationales in the target language, so keyword signals will be
        missed for non-Latin-script languages. 'judge' is acceptable but its
        rationales are synthesis text rather than independent model reasoning.
    norm_threshold : float
        Fraction of mean string length used as edit-distance tolerance
        for MEASUREMENT_ARTEFACT detection.
    output_dir : str, optional
        Defaults to translated_terms/{first_term}/evaluation/

    Returns
    -------
    pd.DataFrame with one row per (language_code × term_source)
    """
    if rationale_variant == 'native_rationale':
        console.print(
            "⚠ 'native_rationale' produces rationales in the target language. "
            "English keyword matching will miss signals for non-Latin-script languages. "
            "Use 'minimal' or 'expert_persona' for reliable keyword analysis.",
            style="bold yellow"
        )

    tokens = source_tokens if source_tokens is not None else SOURCE_TOKENS
    all_rows = []

    for term in target_terms:
        term_slug = term.lower().replace(' ', '_')
        console.print(f"\n[bold cyan]Analyzing disagreements: {term}[/bold cyan]")

        # Load across_variant_detail for best_candidates per service
        eval_dir = os.path.join(
            data_directory_path, 'translated_terms', term_slug, 'evaluation'
        )
        detail_path = os.path.join(eval_dir, 'across_variant_detail.csv')
        if not os.path.exists(detail_path):
            console.print(
                f"  ⚠ across_variant_detail.csv not found — run "
                "analyze_confidence_across_variants.py first",
                style="bold yellow"
            )
            continue

        detail_df = read_csv_file(detail_path)
        detail_df = detail_df[detail_df['term_source'] == term]

        # Load the reference variant DataFrame for rationale columns
        variant_df = load_variant_df(data_directory_path, term_slug, rationale_variant)
        if variant_df is None:
            console.print(
                f"  ⚠ Could not load '{rationale_variant}' variant data",
                style="bold yellow"
            )
            variant_df = pd.DataFrame()

        console.print(
            f"  detail rows: {len(detail_df)} | "
            f"variant rows: {len(variant_df)}"
        )

        # Index variant_df for fast lookup
        if not variant_df.empty:
            variant_df = variant_df.set_index(['language_code', 'term_source'])

        for (lang_code, term_source), grp in detail_df.groupby(
            ['language_code', 'term_source']
        ):
            lang_name = grp['language_name'].iloc[0]
            lang_family = grp['language_family'].iloc[0]

            # Build translations dict: service → best_candidate across variants
            translations: Dict[str, Optional[str]] = {}
            for _, srow in grp.iterrows():
                candidate = srow.get('best_candidate')
                translations[srow['service']] = (
                    str(candidate) if pd.notna(candidate) else None
                )

            # Check Wikipedia
            wiki_row = grp[grp['service'] == 'Wikipedia']
            has_wikipedia = (
                not wiki_row.empty
                and pd.notna(wiki_row.iloc[0].get('best_candidate'))
                and str(wiki_row.iloc[0].get('best_candidate', '')).strip() not in ('', 'nan')
            )

            # Extract rationales from variant_df
            rationales: Dict[str, Optional[str]] = {}
            if not variant_df.empty:
                key = (lang_code, term_source)
                if key in variant_df.index:
                    vrow = variant_df.loc[key]
                    if isinstance(vrow, pd.DataFrame):
                        vrow = vrow.iloc[0]
                    for service, col in RATIONALE_COLS.items():
                        val = vrow.get(col) if col in vrow.index else None
                        rationales[service] = str(val) if pd.notna(val) else None

            result = classify_language(
                translations=translations,
                rationales=rationales,
                has_wikipedia=has_wikipedia,
                norm_threshold=norm_threshold,
                source_tokens=tokens,
            )

            rationale_similarity = compute_rationale_similarity(rationales)

            # Build the output row
            row: Dict = {
                'language_code': lang_code,
                'language_name': lang_name,
                'language_family': lang_family,
                'term_source': term_source,
                'has_wikipedia': has_wikipedia,
                'n_services_with_data': sum(1 for t in translations.values() if t),
                'mean_rationale_similarity': rationale_similarity,
            }
            row.update(result)

            # Flatten signal lists to comma-separated strings for CSV
            row['absence_signals'] = '; '.join(
                f"{s}:{kw}" for s, kw in result.get('absence_signals', [])
            )
            row['construction_signals'] = '; '.join(
                f"{s}:{kw}" for s, kw in result.get('construction_signals', [])
            )
            row['debate_signals'] = '; '.join(
                f"{s}:{kw}" for s, kw in result.get('debate_signals', [])
            )
            row['has_script_disagreement'] = result.get('has_script_disagreement', False)
            row['script_signals'] = '; '.join(
                f"{s}:{sc}" for s, sc in result.get('script_per_service', [])
            )

            # Attach first 300 chars of each rationale for inspection
            for service in ('Claude', 'OpenAI', 'Gemini', 'Ollama'):
                text = rationales.get(service, '')
                row[f'{service.lower().replace(" ", "_")}_rationale_snippet'] = (
                    text[:300] if text else ''
                )

            # All service translations for reference
            row['service_translations'] = str({
                s: t for s, t in translations.items() if t
            })

            all_rows.append(row)

    if not all_rows:
        console.print("⚠ No rows to classify.", style="bold red")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_rows)

    # ── Summary ───────────────────────────────────────────────────────────
    cat_counts = result_df['category'].value_counts()
    _print_summary(result_df, cat_counts)

    # ── Output ────────────────────────────────────────────────────────────
    _dir = output_dir or os.path.join(
        data_directory_path, 'translated_terms',
        target_terms[0].lower().replace(' ', '_'), 'evaluation'
    )
    os.makedirs(_dir, exist_ok=True)
    out_path = os.path.join(_dir, 'disagreement_analysis.csv')
    result_df.to_csv(out_path, index=False)
    console.print(f"\n[bold green]✓ Wrote {len(result_df)} rows → {out_path}[/bold green]")

    return result_df


def _print_summary(df: pd.DataFrame, cat_counts: pd.Series) -> None:
    table = Table(title="Disagreement Classification Summary", show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="white")
    table.add_column("Pct", style="white")
    table.add_column("Description", style="dim")
    total = len(df)
    for cat, count in cat_counts.items():
        table.add_row(
            cat,
            str(count),
            f"{count / total * 100:.1f}%",
            CATEGORIES.get(cat, ''),
        )
    console.print(table)

    # Per-family breakdown for the interesting categories
    interesting = ['PRODUCTIVE_DISAGREEMENT', 'STRUCTURAL_ABSENCE', 'TRANSMOGRIFICATION']
    family_counts = (
        df[df['category'].isin(interesting)]
        .groupby(['language_family', 'category'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if not family_counts.empty:
        console.print("\n[bold]Category × Language Family[/bold]")
        console.print(family_counts.to_string(index=False))


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Classify translation disagreements into four uncertainty categories."
    )
    parser.add_argument('--term', nargs='+', default=['Digital Humanities'])
    parser.add_argument(
        '--variant', default='minimal',
        choices=['minimal', 'expert_persona', 'native_rationale', 'judge'],
        help='Prompt variant to use for rationale extraction (default: minimal)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.15,
        help='Edit-distance tolerance for artefact detection (default: 0.15 × mean length)'
    )
    parser.add_argument(
        '--source-tokens', nargs='+', default=None,
        help=(
            'Loan-word tokens to detect (default: Digital Humanities tokens). '
            'Pass space-separated values, e.g. --source-tokens digital humanities dh'
        )
    )
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    data_dir = get_data_directory_path()
    run_disagreement_analysis(
        data_directory_path=data_dir,
        target_terms=args.term,
        rationale_variant=args.variant,
        norm_threshold=args.threshold,
        source_tokens=args.source_tokens,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
