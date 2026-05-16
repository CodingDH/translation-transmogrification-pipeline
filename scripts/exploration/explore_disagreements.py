#!/usr/bin/env python3
"""
explore_disagreements.py
========================
Classify translation disagreements into five uncertainty categories:

1. COMPLETE_CONSENSUS: All services agree. No disagreement to classify.
2. MEASUREMENT_ARTEFACT: Apparent disagreement dissolves under normalization or is entirely explained by writing-script variation (e.g. Serbian Cyrillic ↔ Latin, Esperanto Ciferecaj vs Cifereca).
3. PRODUCTIVE_DISAGREEMENT: Multiple legitimate in-language alternatives. Wikipedia presence is the positive signal: community presence, proxied by Wikipedia, reframes multi-output disagreement as translation-strategy debate rather than structural absence.
4. STRUCTURAL_ABSENCE: The concept has no community equivalent. Detected by loan-word borrowing (services use the source term unadapted), very sparse coverage, or convergent rationale signals admitting absence.
5. TRANSMOGRIFICATION: Default confident fluent output with no community anchor. The model produced a word-like object with no community practice to verify against.

Classification hierarchy (applied in order, first rule whose predicate fires
wins, see CLASSIFIER_RULES below for the canonical list):

  1. No translations at all                → STRUCTURAL_ABSENCE
  2. All services agree                    → COMPLETE_CONSENSUS
  3. Script-only disagreement              → MEASUREMENT_ARTEFACT
  4. Normalization collapses disagreement  → MEASUREMENT_ARTEFACT
  5. Wikipedia present + multiple outputs  → PRODUCTIVE_DISAGREEMENT
  6. Loan-word present                     → STRUCTURAL_ABSENCE
  7. Sparse coverage + absence keywords    → STRUCTURAL_ABSENCE  *(keyword)
  8. Convergent absence keywords           → STRUCTURAL_ABSENCE  *(keyword)
  9. Default                               → TRANSMOGRIFICATION

Rules marked *(keyword) read rationale text for English-language signal words. They are gated on `rationales_are_english` and no-op when the rationale variant is in the target language. They can additionally be disabled entirely with `--no-keyword-rules` for ablation experiments. When turned off, the classifier
runs on string/metadata features only, and the `rule_fired` column records which rule produced the verdict for every language so you can audit the ablation's effect row-by-row.

The load-bearing ordering decision is that PRODUCTIVE_DISAGREEMENT (Wikipedia presence) fires before STRUCTURAL_ABSENCE (loan-word presence). A language with an established scholarly community whose services sometimes borrow the source term is showing translation-strategy debate, not structural absence. Reversing the order would re-classify German, Swedish, and other high-resource European languages as STRUCTURAL_ABSENCE any time one service output "Digital Humanities" untranslated — contradicting direct evidence of active DH communities there.

Data quality
------------
LLMs occasionally return a translation but omit the rationale, or return a
placeholder string such as "No rationale provided" instead of real reasoning.
Both cases are treated as missing rationales:

  - ``load_variant_df`` (explore_confidence_within_variant.py) nulls out any
    translation whose rationale is absent or a placeholder, and vice versa.
    This pairing is enforced before any notebook or script sees the data.
  - Within this script, the same ``is_placeholder_rationale`` check is applied
    a second time when building the per-language rationale dict from the variant
    DataFrame index.
  - After both dicts are built, any LLM service that still has a translation
    but no rationale in the loaded variant is dropped from the ``translations``
    dict, so its output does not influence the classifier or rationale-similarity
    score.

Input
-----
  - translated_terms/{term}/evaluation/across_variant_detail.csv
    (produced by explore_confidence_across_variants.py)
  - translated_terms/{term}/{direct,prompt}_services/*.csv
    loaded via load_variant_df for a single reference variant's rationales

Output
------
  translated_terms/{term}/evaluation/disagreement_analysis.csv
    One row per (language_code × term_source) with:
      - category, rule_fired, evidence
      - classifier feature columns (unique counts, edit distance, script info,
        loan words, signal counts)
      - mean_rationale_similarity (surface lexical, not semantic — see docs)
      - first 300 chars of each service's rationale for inspection

When --no-keyword-rules is passed, output is written to
disagreement_analysis_no_keywords.csv so ablation runs don't overwrite the
canonical output.

Usage
-----
    # Default: all rules on, canonical classification
    python explore_disagreements.py

    # Single term, non-default variant
    python explore_disagreements.py --term "Digital Humanities" --variant expert_persona

    # Ablation: keyword-rule contribution
    python explore_disagreements.py --no-keyword-rules
    # Then diff rule_fired between the two CSVs.

    # Override source tokens for a different term
    python explore_disagreements.py --source-tokens digital humanities dh
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.utils import (
    get_data_directory_path, read_csv_file,
    detect_script_disagreement,
    is_placeholder_rationale,
)
from scripts.exploration.explore_confidence_within_variant import load_variant_df

console = Console()


# ── Constants ──────────────────────────────────────────────────────────────

CATEGORIES = {
    'COMPLETE_CONSENSUS':      'All services agree — no disagreement to classify',
    'MEASUREMENT_ARTEFACT':    'Disagreement dissolves after normalization',
    'PRODUCTIVE_DISAGREEMENT': 'Multiple legitimate in-language alternatives',
    'STRUCTURAL_ABSENCE':      'Concept absent from community; model borrows or signals absence',
    'TRANSMOGRIFICATION':      'Confident fluent output untethered from community practice',
}

# Which prompt variants produce English-language rationales. Keyword matching assumes English; native_rationale produces rationales in the target language and keyword rules are skipped for it.
ENGLISH_RATIONALE_VARIANTS = {'minimal', 'expert_persona', 'judge'}
NATIVE_RATIONALE_VARIANTS = {'native_rationale'}

RATIONALE_COLS = {
    'Claude':   'claude_translation_rationale',
    'OpenAI':   'openai_translation_rationale',
    'Gemini':   'gemini_translation_rationale',
    'DeepSeek': 'deepseek_translation_rationale',
    'Llama':    'llama_translation_rationale',
    'Gemma':    'gemma_translation_rationale',
    'Qwen':     'qwen_translation_rationale',
    'Mistral':  'mistral_translation_rationale',
}

TRANSLATION_COLS = {
    'Wikipedia':        'wikipedia_translated_term',
    'Google Translate': 'gt_translated_term',
    'EasyNMT':          'enmt_translated_term',
    'Lingvanex':        'lingvanex_translated_term',
    'OpenAI':           'openai_translated_term',
    'Claude':           'claude_translated_term',
    'Gemini':           'gemini_translated_term',
    'DeepSeek':         'deepseek_translated_term',
    'Llama':            'llama_translated_term',
    'Gemma':            'gemma_translated_term',
    'Qwen':             'qwen_translated_term',
    'Mistral':          'mistral_translated_term',
}


# Rationale keywords signaling the model knows no equivalent exists. Literal-word patterns use \b word boundaries so "overborrowed" or "colloquialism" don't accidentally match "borrow" or "anglicism".
ABSENCE_KEYWORDS = [
    'no equivalent', 'no direct', 'no established', 'no widely', 'not adopted',
    'not institutionalized', 'not institutionalised', 'not yet adopted',
    'does not exist', "hasn't been", 'have not been', 'has not been',
    'no community', 'no scholarly', 'not used in', 'concept does not',
    'term does not', 'no translation', 'untranslated', 'no native',
    'no recognized', 'no recognised',
    r'\bborrowed\b', r'\bloanword\b', r'\bloan word\b',
    r'\bborrowing\b', r'\banglicism\b', r'\banglicization\b', r'\banglicised\b',
    r'\btransliterat', r'keep.*english', r'retain.*english',
]

# Rationale keywords suggesting elaborate construction as a transmogrification signal. Currently recorded in the output CSV but not used by any rule in the chain; retained because the signal is potentially useful for downstream analysis.
CONSTRUCTION_KEYWORDS = [
    'etymolog', 'root word', 'compound', 'combining', 'derive from',
    'ancient', 'classical', 'proto-', 'reconstructed', 'neologism',
    'coined', 'coining', 'create.*term', 'construct', 'formation',
    'morpholog', 'word-forming', 'suffix', 'prefix',
]

# Rationale keywords suggesting community debate as a productive disagreement signal. Also recorded but not used by any rule (Wikipedia presence does the work).
DEBATE_KEYWORDS = [
    'some scholars', 'different communities', 'preferred by', 'others use',
    'alternatively', 'debate', 'contested', 'competing', 'variant',
    'both terms', 'either', 'also used', 'sometimes rendered',
]

def derive_source_tokens(term: str) -> List[str]:
    """Derive loanword-detection tokens from the source term itself.

    Three forms, in order:
      1. Full lowercased term          — catches exact phrase borrowing
      2. Individual words              — catches per-word borrowing
      3. Initial-letter acronym        — catches abbreviated forms (e.g. 'dh')

    This replaces manual curation: tokens emerge from the term itself so no
    domain knowledge is required when adding a new source term.
    
    Parameters
    ----------
    term : str
		The source term, e.g. "Digital Humanities". Should be in English or
		Latin script for the tokenization to work well; non-Latin scripts are
		not guaranteed to split into meaningful tokens but will still be
		included as-is in the output list so the loan-word rule can still fire
		on exact matches.
        
    Returns
    -------
    List[str]
		A list of tokens derived from the term, e.g. ["digital humanities", "digital", "humanities", "dh"].
    """
    term_lower = term.lower().strip()
    words = [w for w in re.split(r'\s+', term_lower) if w]
    tokens = [term_lower] + words
    if len(words) > 1:
        tokens.append(''.join(w[0] for w in words))
    seen: set = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ── String normalization ───────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip diacritics, remove punctuation, collapse whitespace.

    RTL scripts (Arabic, Hebrew) are handled correctly: Python's \\w is
    Unicode-aware so Arabic/Hebrew letters are preserved by the punctuation
    strip, and NFKD decomposition correctly normalises Arabic compatibility
    forms. The only combining characters removed are vowel-pointing marks
    (harakat, niqqud) which is a loss but necessary to avoid counting diacritic variation as disagreement. Though we do count it as disagreement at the raw level, so the edit-distance-based rules can still detect it as a measurement artefact when the underlying term is the same.
    """
    if not text or not isinstance(text, str):
        return ''
    nfkd = unicodedata.normalize('NFKD', text.lower())
    no_combining = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', no_combining)).strip()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance. Returns 0 for identical strings, 1 for one insertion/deletion/substitution, etc."""
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
    return max(
        edit_distance(a, b)
        for i, a in enumerate(strings)
        for b in strings[i + 1:]
    )


def find_loan_words(text: str, source_tokens: List[str]) -> List[str]:
    """Return the subset of source_tokens present as whole words in text.

    Uses whole-word matching (post-normalization) so adapted inflections like 'digitala' (Gothic inflection of 'digital') are not flagged but instead only unadapted borrowings like 'digital' or 'humanities' as standalone words.
    """
    if not text:
        return []
    words = set(normalize(text).split())
    return [tok for tok in source_tokens if tok in words]


# ── Rationale similarity (surface metric) ─────────────────────────────────

def compute_rationale_similarity(
    rationales: Dict[str, Optional[str]],
) -> Optional[float]:
    """Mean pairwise TF-IDF cosine similarity across service rationales.

    Uses character n-grams (3–5) so the metric is script-agnostic. This is a SURFACE metric: it measures how similarly two rationales are *phrased*, not whether they convey the same meaning. High similarity can arise from shared topic vocabulary even when the underlying claims differ. This similarity therefore is lexical convergence, not semantic agreement.

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

def match_keywords(text: str, keywords: List[str]) -> List[str]:
    """Return which keyword patterns (as regex) match in text."""
    if not text or not isinstance(text, str):
        return []
    text_lower = text.lower()
    return [kw for kw in keywords if re.search(kw, text_lower)]


def extract_rationale_signals(
    rationales: Dict[str, Optional[str]],
) -> Dict[str, List[Tuple[str, str]]]:
    """Scan all service rationales for absence / construction / debate keywords.

    Returns dict with keys 'absence', 'construction', 'debate', each a list
    of (service, keyword) tuples matched.
    """
    signals: Dict[str, List[Tuple[str, str]]] = {
        'absence': [], 'construction': [], 'debate': [],
    }
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


# ── Features + rules ──────────────────────────────────────────────────────

@dataclass
class ClassifierConfig:
    """Tunable knobs for the classifier, in one place."""
    norm_threshold: float = 0.15
    min_distinct_for_pd: int = 2
    sparse_coverage_max: int = 2
    min_absence_signals: int = 2
    source_tokens: Optional[Tuple[str, ...]] = None  # None = derive from term at runtime
    use_keyword_rules: bool = True


@dataclass
class Features:
    """All features a rule might read, computed once per (language × term).

    This is a plain data record of the rules read fields but it never mutates them. Keeps the rule functions short and makes it easy to add new rules without re-deriving features inside each one.
    
    The features are all derived from the raw translation outputs and rationales, plus Wikipedia presence as external metadata. The rationale-based features are gated on `rationales_are_english` because the keyword matching only works for English rationales; when False, the rationale fields are still populated in the output CSV for inspection but the keyword-rule functions are no-ops.
    """
    translations: Dict[str, str]
    rationales: Dict[str, str]
    has_wikipedia: bool
    rationales_are_english: bool
    n_unique_raw: int
    n_unique_normalized: int
    max_edit_distance: int
    dist_threshold: int
    script_info: Dict
    loan_words_found: List[str]
    signals: Dict[str, List[Tuple[str, str]]]


def extract_features(
    translations: Dict[str, Optional[str]],
    rationales: Dict[str, Optional[str]],
    has_wikipedia: bool,
    rationales_are_english: bool,
    config: ClassifierConfig,
) -> Features:
    """Compute all classifier features in one pass. See Features for fields."""
    valid = {
        s: t.strip() for s, t in translations.items()
        if t and isinstance(t, str) and t.strip() and t not in ('nan', 'None')
    }
    valid_rationales = {
        s: t.strip() for s, t in rationales.items()
        if t and isinstance(t, str) and t.strip() and t not in ('nan', 'None')
    } if rationales_are_english else {}

    raw_vals = list(valid.values())
    unique_raw = set(v.strip().lower() for v in raw_vals)
    norm_vals = [normalize(v) for v in raw_vals]
    unique_norm = set(v for v in norm_vals if v)
    norm_list = list(unique_norm)
    max_dist = max_pairwise_edit_distance(norm_list)
    mean_len = (
        sum(len(v) for v in norm_list) / len(norm_list) if norm_list else 1
    )
    dist_threshold = max(2, int(config.norm_threshold * mean_len))

    script_info = detect_script_disagreement(valid)

    # Loan words: union across all services' outputs
    loan_set: set = set()
    for t in valid.values():
        loan_set.update(find_loan_words(t, list(config.source_tokens)))
    loan_words_found = sorted(loan_set)

    # Keyword signals — only extract when rationales are English
    if rationales_are_english:
        signals = extract_rationale_signals(valid_rationales)
    else:
        signals = {'absence': [], 'construction': [], 'debate': []}

    return Features(
        translations=valid,
        rationales=valid_rationales,
        has_wikipedia=has_wikipedia,
        rationales_are_english=rationales_are_english,
        n_unique_raw=len(unique_raw),
        n_unique_normalized=len(unique_norm),
        max_edit_distance=max_dist,
        dist_threshold=dist_threshold,
        script_info=script_info,
        loan_words_found=loan_words_found,
        signals=signals,
    )


# ── Rules ──────────────────────────────────────────────────────────────────
#
# Each rule is a pure function (Features, ClassifierConfig) → Optional[tuple]. Return None to skip; return (category, evidence) to fire and stop the chain. The name of the function that fires is recorded as rule_fired in the output CSV so you can audit which rules do what work.

Rule = Callable[[Features, ClassifierConfig], Optional[Tuple[str, str]]]


def rule_no_translations(f: Features, cfg: ClassifierConfig):
    """Empty translation set → STRUCTURAL_ABSENCE."""
    if not f.translations:
        return ('STRUCTURAL_ABSENCE', 'No service produced a translation')
    return None


def rule_all_agree(f: Features, cfg: ClassifierConfig):
    """Every service produced the same string → COMPLETE_CONSENSUS."""
    if f.n_unique_raw == 1:
        return ('COMPLETE_CONSENSUS', 'All services produced the same translation')
    return None


def rule_script_only_disagreement(f: Features, cfg: ClassifierConfig):
    """Disagreement is entirely explained by writing script."""
    if f.script_info.get('has_script_disagreement') and f.n_unique_normalized <= 1:
        scripts = '+'.join(f.script_info.get('scripts_found', []))
        return ('MEASUREMENT_ARTEFACT',
                f'Script-only disagreement ({scripts}); '
                f'same term in {f.n_unique_raw} scripts')
    return None


def rule_normalization_artifact(f: Features, cfg: ClassifierConfig):
    """Disagreement dissolves under case/diacritic/edit-distance tolerance."""
    if f.n_unique_normalized <= 1 or f.max_edit_distance <= f.dist_threshold:
        return ('MEASUREMENT_ARTEFACT',
                f'Normalized to {f.n_unique_normalized} unique value(s); '
                f'max edit distance {f.max_edit_distance} ≤ threshold {f.dist_threshold}')
    return None


def rule_wikipedia_presence(f: Features, cfg: ClassifierConfig):
    """Wikipedia exists + multiple distinct outputs → PRODUCTIVE_DISAGREEMENT.

    Fires BEFORE loan-word detection by design — see module docstring.
    """
    if f.has_wikipedia and f.n_unique_raw >= cfg.min_distinct_for_pd:
        return ('PRODUCTIVE_DISAGREEMENT',
                f'Wikipedia present; {f.n_unique_raw} distinct outputs; '
                f'debate signals: {len(f.signals["debate"])}')
    return None


def rule_loanword_absence(f: Features, cfg: ClassifierConfig):
    """At least one service borrowed the source term unadapted.

    Reads translations only — works regardless of rationale language.
    """
    if f.loan_words_found:
        return ('STRUCTURAL_ABSENCE',
                f'Loan words present: {f.loan_words_found}')
    return None


def rule_sparse_coverage_absence(f: Features, cfg: ClassifierConfig):
    """Very few services produced output AND rationales admit absence.

    Keyword-dependent: requires English rationales. Disabled by --no-keyword-rules.
    """
    if not cfg.use_keyword_rules or not f.rationales_are_english:
        return None
    if (len(f.translations) <= cfg.sparse_coverage_max
            and len(f.signals['absence']) >= cfg.min_absence_signals):
        return ('STRUCTURAL_ABSENCE',
                f'Sparse coverage ({len(f.translations)} services) + '
                f'{len(f.signals["absence"])} absence signals')
    return None


def rule_convergent_absence(f: Features, cfg: ClassifierConfig):
    """Services converge on few outputs AND rationales admit absence.

    Keyword-dependent: requires English rationales. Disabled by --no-keyword-rules.
    """
    if not cfg.use_keyword_rules or not f.rationales_are_english:
        return None
    if f.n_unique_raw <= 2 and len(f.signals['absence']) >= cfg.min_absence_signals:
        return ('STRUCTURAL_ABSENCE',
                f'Convergent absence ({len(f.signals["absence"])} signals) '
                f'across {f.n_unique_raw} distinct outputs')
    return None


def rule_default_transmogrification(f: Features, cfg: ClassifierConfig):
    """Catch-all: no Wikipedia, no borrowing, no absence signal → TRANSMOGRIFICATION."""
    return ('TRANSMOGRIFICATION',
            f'No Wikipedia; {f.n_unique_raw} distinct outputs; '
            f'construction signals: {len(f.signals["construction"])}')


# The canonical rule chain. The ORDER encodes the classifier's theory: measurement artefacts come out first, then Wikipedia-gated productive disagreement, then absence detection (loanword before keywords), then transmogrification as the default.
CLASSIFIER_RULES: Tuple[Rule, ...] = (
    rule_no_translations,
    rule_all_agree,
    rule_script_only_disagreement,
    rule_normalization_artifact,
    rule_wikipedia_presence,
    rule_loanword_absence,
    rule_sparse_coverage_absence,    # keyword-dependent
    rule_convergent_absence,         # keyword-dependent
    rule_default_transmogrification,
)


def classify(
    features: Features,
    config: ClassifierConfig,
    rules: Tuple[Rule, ...] = CLASSIFIER_RULES,
) -> Tuple[str, str, str]:
    """Run the rule chain until a rule fires.

    Returns (category, evidence, rule_name). The rule_name is the __name__ of
    the function that fired, recorded in the CSV so the ablation can be
    audited row-by-row.
    """
    for rule in rules:
        out = rule(features, config)
        if out is not None:
            category, evidence = out
            return category, evidence, rule.__name__
    # rule_default_transmogrification always fires, so we should never reach here
    raise RuntimeError(
        "Rule chain exhausted without a match. "
        "CLASSIFIER_RULES must end with a catch-all."
    )


# ── Main pipeline ──────────────────────────────────────────────────────────

_RATIONALE_VARIANT_KEYS = [
    'rationale_minimal',
    'rationale_expert_persona',
    'rationale_native_rationale',
    'rationale_judge',
]

_TERM_VARIANT_KEYS = [
    'term_minimal',
    'term_expert_persona',
    'term_native_rationale',
    'term_judge',
]


def load_exclusions(path: str) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """Load manual_exclusions.csv into a nested dict.

    Returns ``{language_code: {service: {term: bool, rationale_minimal: bool, ...}}}``.
    The file is produced by html_files/disagreement_explorer.html.
    Per-variant rationale keys: rationale_minimal, rationale_expert_persona,
    rationale_native_rationale, rationale_judge.
    """
    df = pd.read_csv(path, converters={'language_code': str})
    result: Dict[str, Dict[str, Dict[str, bool]]] = {}
    for _, row in df.iterrows():
        code = str(row.get('language_code', '')).strip()
        svc  = str(row.get('service', '')).strip()
        if not code or not svc:
            continue
        if code not in result:
            result[code] = {}
        entry: Dict[str, bool] = {
            'term': str(row.get('exclude_translation', 'False')).strip().lower() == 'true',
        }
        for vkey in _TERM_VARIANT_KEYS:
            col = f'exclude_{vkey}'
            entry[vkey] = str(row.get(col, 'False')).strip().lower() == 'true'
        # Auto-promote: if all per-variant terms are excluded, treat as full exclusion
        if not entry['term'] and all(entry.get(k, False) for k in _TERM_VARIANT_KEYS):
            entry['term'] = True
        for vkey in _RATIONALE_VARIANT_KEYS:
            col = f'exclude_{vkey}'
            entry[vkey] = str(row.get(col, 'False')).strip().lower() == 'true'
        result[code][svc] = entry
    n_langs = len(result)
    n_entries = sum(len(v) for v in result.values())
    console.print(
        f"  Loaded {n_entries} exclusion entries across {n_langs} languages from {path}",
        style="dim cyan",
    )
    return result


def load_term_corrections(path: str) -> Dict[str, Dict[str, str]]:
    """Load term corrections from manual_exclusions.csv.

    Returns ``{language_code: {original_term: corrected_term}}`` for rows
    where ``service == 'term_correction'``.
    """
    df = pd.read_csv(path, converters={'language_code': str})
    result: Dict[str, Dict[str, str]] = {}
    if 'corrected_term' not in df.columns or 'original_term' not in df.columns:
        return result
    for _, row in df.iterrows():
        if str(row.get('service', '')).strip() != 'term_correction':
            continue
        code = str(row.get('language_code', '')).strip()
        original = str(row.get('original_term', '')).strip()
        corrected = str(row.get('corrected_term', '')).strip()
        if code and original and corrected and corrected not in ('', 'nan', 'None'):
            if code not in result:
                result[code] = {}
            result[code][original] = corrected
    n_corrections = sum(len(v) for v in result.values())
    if n_corrections:
        console.print(
            f"  Loaded {n_corrections} term correction(s) across {len(result)} language(s) from {path}",
            style="dim cyan",
        )
    return result


def run_disagreement_analysis(
    data_directory_path: str,
    target_terms: List[str],
    rationale_variant: str = 'minimal',
    config: Optional[ClassifierConfig] = None,
    output_dir: Optional[str] = None,
    exclusions: Optional[Dict[str, Dict[str, Dict[str, bool]]]] = None,
    corrections: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """Classify every (language × term) disagreement.

    Parameters
    ----------
    data_directory_path : str
        Root data directory.
    target_terms : list of str
        Each term needs an across_variant_detail.csv from
        explore_confidence_across_variants.py.
    rationale_variant : str
        Which prompt variant to source rationales from. 'minimal' is the
        canonical choice. 'native_rationale' silently disables keyword rules
        because its rationales are in the target language. 'judge' emits a
        warning — its rationales are synthesis text, not independent reasoning,
        so keyword signals mix object-level and meta-level evidence.
    config : ClassifierConfig, optional
        Classifier knobs (thresholds, source tokens, keyword-rule toggle).
        Defaults to ClassifierConfig() with source tokens tuned for Digital
        Humanities.
    corrections : dict, optional
        Term corrections from load_term_corrections(). Maps
        {language_code: {original_term: corrected_term}}. Applied after
        exclusions, replacing matched translation strings before classification.
    output_dir : str, optional
        Defaults to translated_terms/{first_term}/evaluation/.

    Returns
    -------
    pd.DataFrame with one row per (language_code × term_source).
    """
    cfg = config or ClassifierConfig()

    if rationale_variant in NATIVE_RATIONALE_VARIANTS:
        console.print(
            "ℹ Running with native_rationale variant. Keyword-dependent "
            "rules will no-op; classification will use string/metadata features only.",
            style="dim cyan",
        )
        rationales_are_english = False
    elif rationale_variant in ENGLISH_RATIONALE_VARIANTS:
        if rationale_variant == 'judge':
            console.print(
                "⚠ Using 'judge' variant: its rationales are synthesis text "
                "across other variants, not independent model reasoning. "
                "Keyword signals may conflate meta-level and object-level "
                "evidence. Consider 'minimal' for the canonical classification.",
                style="bold yellow",
            )
        rationales_are_english = True
    else:
        raise ValueError(
            f"Unknown rationale_variant: {rationale_variant!r}. "
            f"Expected one of {ENGLISH_RATIONALE_VARIANTS | NATIVE_RATIONALE_VARIANTS}."
        )

    console.print(f"\n[bold]Classifier configuration[/bold]")
    console.print(f"  rationale_variant        = {rationale_variant}")
    console.print(f"  rationales_are_english   = {rationales_are_english}")
    console.print(f"  use_keyword_rules        = {cfg.use_keyword_rules}")
    console.print(f"  norm_threshold           = {cfg.norm_threshold}")
    console.print(f"  min_distinct_for_pd      = {cfg.min_distinct_for_pd}")
    console.print(f"  min_absence_signals      = {cfg.min_absence_signals}")
    console.print(f"  source_tokens            = {'(derived from term)' if cfg.source_tokens is None else list(cfg.source_tokens)}")

    all_rows: List[Dict] = []

    for term in target_terms:
        term_slug = term.lower().replace(' ', '_')
        console.print(f"\n[bold cyan]Analyzing disagreements: {term}[/bold cyan]")
        term_tokens = cfg.source_tokens if cfg.source_tokens is not None else tuple(derive_source_tokens(term))
        cfg_term = replace(cfg, source_tokens=term_tokens)
        console.print(f"  source_tokens: {list(term_tokens)}")

        eval_dir = os.path.join(
            data_directory_path, 'translated_terms', term_slug, 'evaluation',
        )
        detail_path = os.path.join(eval_dir, 'across_variant_detail.csv')
        if not os.path.exists(detail_path):
            console.print(
                f"  ⚠ {detail_path} not found — run "
                "explore_confidence_across_variants.py first",
                style="bold yellow",
            )
            continue

        detail_df = read_csv_file(detail_path)
        detail_df = detail_df[detail_df['term_source'] == term]

        variant_df = load_variant_df(data_directory_path, term_slug, rationale_variant)
        if variant_df is None:
            console.print(
                f"  ⚠ Could not load '{rationale_variant}' variant data; "
                "rationales will be empty",
                style="bold yellow",
            )
            variant_df = pd.DataFrame()

        console.print(
            f"  detail rows: {len(detail_df)} | variant rows: {len(variant_df)}",
        )
        if not variant_df.empty:
            variant_df = variant_df.set_index(['language_code', 'term_source'])

        for (lang_code, term_source), grp in detail_df.groupby(
            ['language_code', 'term_source'],
            dropna=False,
        ):
            lang_name = grp['language_name'].iloc[0]
            lang_family = grp['language_family'].iloc[0]

            translations: Dict[str, Optional[str]] = {}
            for _, srow in grp.iterrows():
                candidate = srow.get('best_candidate')
                translations[srow['service']] = (
                    str(candidate) if pd.notna(candidate) else None
                )

            # Apply manual exclusions from html_files/disagreement_explorer.html
            if exclusions:
                lang_excl = exclusions.get(lang_code, {})
                translations = {
                    s: t for s, t in translations.items()
                    if not lang_excl.get(s, {}).get('term', False)
                }

            # Apply term corrections: replace original translation strings with
            # user-corrected values before classification
            if corrections:
                lang_corr = corrections.get(lang_code, {})
                if lang_corr:
                    translations = {
                        s: (lang_corr[t] if t and t in lang_corr else t)
                        for s, t in translations.items()
                    }

            wiki_row = grp[grp['service'] == 'Wikipedia']
            has_wikipedia = (
                not wiki_row.empty
                and pd.notna(wiki_row.iloc[0].get('best_candidate'))
                and str(wiki_row.iloc[0].get('best_candidate', '')).strip()
                    not in ('', 'nan')
            )

            rationales: Dict[str, Optional[str]] = {}
            if not variant_df.empty:
                key = (lang_code, term_source)
                if key in variant_df.index:
                    vrow = variant_df.loc[key]
                    if isinstance(vrow, pd.DataFrame):
                        vrow = vrow.iloc[0]
                    for service, col in RATIONALE_COLS.items():
                        val = vrow.get(col) if col in vrow.index else None
                        if pd.notna(val) and not is_placeholder_rationale(str(val)):
                            rationales[service] = str(val)
                        else:
                            rationales[service] = None

            if exclusions:
                lang_excl = exclusions.get(lang_code, {})
                variant_key = f'rationale_{rationale_variant}'
                rationales = {
                    s: r for s, r in rationales.items()
                    if not lang_excl.get(s, {}).get(variant_key, False)
                }

            # Drop LLM translations that have no rationale in this variant.
            # Arises when across_variant_detail pulled a translation from a
            # different variant file than the one loaded for rationales.
            _LLM_SVCS = {'Claude', 'OpenAI', 'Gemini', 'Ollama'}
            translations = {
                s: t for s, t in translations.items()
                if s not in _LLM_SVCS or rationales.get(s) is not None
            }

            features = extract_features(
                translations=translations,
                rationales=rationales,
                has_wikipedia=has_wikipedia,
                rationales_are_english=rationales_are_english,
                config=cfg_term,
            )
            category, evidence, rule_name = classify(features, cfg_term)
            similarity = compute_rationale_similarity(rationales)

            row: Dict = {
                'language_code': lang_code,
                'language_name': lang_name,
                'language_family': lang_family,
                'term_source': term_source,
                'category': category,
                'rule_fired': rule_name,
                'evidence': evidence,
                'has_wikipedia': has_wikipedia,
                'rationales_are_english': rationales_are_english,
                'n_services_with_translation': len(features.translations),
                'n_services_with_rationale': len(features.rationales),
                'n_unique_raw': features.n_unique_raw,
                'n_unique_normalized': features.n_unique_normalized,
                'max_edit_distance': features.max_edit_distance,
                'loan_words_found': '; '.join(features.loan_words_found),
                'has_script_disagreement': features.script_info.get(
                    'has_script_disagreement', False,
                ),
                'scripts_found': '; '.join(features.script_info.get('scripts_found', [])),
                'absence_signals': '; '.join(
                    f"{s}:{kw}" for s, kw in features.signals['absence']
                ),
                'construction_signals': '; '.join(
                    f"{s}:{kw}" for s, kw in features.signals['construction']
                ),
                'debate_signals': '; '.join(
                    f"{s}:{kw}" for s, kw in features.signals['debate']
                ),
                'mean_rationale_similarity': similarity,
            }
            for service in ('Claude', 'OpenAI', 'Gemini', 'Ollama'):
                text = features.rationales.get(service, '')
                row[f'{service.lower()}_rationale_snippet'] = text[:300] if text else ''

            row['service_translations'] = str(features.translations)
            all_rows.append(row)

    if not all_rows:
        console.print("⚠ No rows to classify.", style="bold red")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_rows)
    _print_summary(result_df)

    _dir = output_dir or os.path.join(
        data_directory_path, 'translated_terms',
        target_terms[0].lower().replace(' ', '_'), 'evaluation',
    )
    os.makedirs(_dir, exist_ok=True)

    # Canonical run (keyword rules on) writes disagreement_analysis.csv.
    # Ablation run (--no-keyword-rules) gets a suffix so it never overwrites
    # the canonical output.
    suffix = '' if cfg.use_keyword_rules else '_no_keywords'
    out_path = os.path.join(_dir, f'disagreement_analysis{suffix}.csv')
    result_df.to_csv(out_path, index=False)
    console.print(f"\n[bold green]✓ Wrote {len(result_df)} rows → {out_path}[/bold green]")

    return result_df


def _print_summary(df: pd.DataFrame) -> None:
    """Print category counts, rule-fired counts, and family breakdown."""
    cat_counts = df['category'].value_counts()
    total = len(df)

    cat_table = Table(title="Disagreement Classification Summary", show_header=True)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Count", style="white")
    cat_table.add_column("Pct", style="white")
    cat_table.add_column("Description", style="dim")
    for cat, count in cat_counts.items():
        cat_table.add_row(
            cat, str(count), f"{count / total * 100:.1f}%",
            CATEGORIES.get(cat, ''),
        )
    console.print(cat_table)

    # Rule-fired counts: which rules actually did work?
    # Useful for ablations — if rule_sparse_coverage_absence never fires,
    # the keyword rule contributed zero independent signal.
    rule_counts = df['rule_fired'].value_counts()
    rule_table = Table(title="Rule-fired counts (which rules did the work)",
                       show_header=True)
    rule_table.add_column("Rule", style="cyan")
    rule_table.add_column("Count", style="white")
    rule_table.add_column("Pct", style="white")
    for rule, count in rule_counts.items():
        rule_table.add_row(
            rule, str(count), f"{count / total * 100:.1f}%",
        )
    console.print(rule_table)

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
        description="Classify translation disagreements into five uncertainty categories.",
    )
    parser.add_argument(
        '--term', nargs='+', default=['Digital Humanities'],
    )
    parser.add_argument(
        '--variant', default='minimal',
        choices=sorted(ENGLISH_RATIONALE_VARIANTS | NATIVE_RATIONALE_VARIANTS),
        help=(
            "Prompt variant for rationale extraction. 'minimal' is canonical; "
            "'native_rationale' disables keyword rules; 'judge' warns."
        ),
    )
    parser.add_argument(
        '--threshold', type=float, default=0.15,
        help='Edit-distance tolerance as fraction of mean length (default 0.15)',
    )
    parser.add_argument(
        '--min-distinct-for-pd', type=int, default=2,
        help='Min distinct outputs for PRODUCTIVE_DISAGREEMENT (default 2)',
    )
    parser.add_argument(
        '--min-absence-signals', type=int, default=2,
        help='Min absence-keyword hits for keyword absence rules (default 2)',
    )
    parser.add_argument(
        '--source-tokens', nargs='+', default=None,
        help=(
            'Override source-language loanword tokens. Defaults to the Digital '
            'Humanities curated list.'
        ),
    )
    parser.add_argument(
        '--no-keyword-rules', action='store_true',
        help=(
            'Disable the two keyword-dependent absence rules '
            '(rule_sparse_coverage_absence, rule_convergent_absence). Output CSV '
            'is suffixed _no_keywords.csv so the ablation run does not overwrite '
            'the canonical output. Diff rule_fired columns against the default '
            'run to see which classifications the keyword rules were providing.'
        ),
    )
    parser.add_argument('--output-dir', default=None)
    parser.add_argument(
        '--exclusions', default=None, metavar='PATH',
        help=(
            'Path to manual_exclusions.csv produced by html_files/disagreement_explorer.html. '
            'Rows specify per-language, per-service translations and/or rationales to remove '
            'before the classifier runs — use this to fix bad translations (e.g. "To be '
            'determined") without editing source data.'
        ),
    )
    args = parser.parse_args()

    source_tokens = (
        tuple(args.source_tokens) if args.source_tokens is not None
        else None  # derive from term at runtime
    )
    cfg = ClassifierConfig(
        norm_threshold=args.threshold,
        min_distinct_for_pd=args.min_distinct_for_pd,
        min_absence_signals=args.min_absence_signals,
        source_tokens=source_tokens,
        use_keyword_rules=not args.no_keyword_rules,
    )

    excl = load_exclusions(args.exclusions) if args.exclusions else None
    corr = load_term_corrections(args.exclusions) if args.exclusions else None

    data_dir = get_data_directory_path()
    run_disagreement_analysis(
        data_directory_path=data_dir,
        target_terms=args.term,
        rationale_variant=args.variant,
        config=cfg,
        output_dir=args.output_dir,
        exclusions=excl,
        corrections=corr,
    )


if __name__ == '__main__':
    main()
