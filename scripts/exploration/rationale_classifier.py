#!/usr/bin/env python3
"""
rationale_classifier.py
=======================
Interpretable text classifiers for studying rationale GENRES across services and
prompt variants. The unit of analysis is the rationale text itself — each row is
one (service × variant × language) rationale — and the question is whether the
classifier can recover service identity or prompt variant from the prose alone.

Two tasks:
  1. Variant classification — predict which prompt variant (minimal,
     expert_persona, judge) produced a rationale. Top log-odds coefficients per
     class surface the rhetorical vocabulary that distinguishes each prompt
     style.
  2. Service classification — predict which LLM service (Claude, OpenAI, Gemini,
     DeepSeek, Llama, Gemma, Qwen, Mistral) produced a rationale. Top
     coefficients are the prose fingerprint of each model's reasoning style.

NOT included as a primary task: a category classifier predicting
STRUCTURAL_ABSENCE / TRANSMOGRIFICATION / etc. from rationale text. That task is
circular when run on rule-based-classifier outputs that themselves used keyword
features in the rationales — the classifier "discovers" exactly the keywords
that produced its labels, plus language-name leakage. The infrastructure here
will technically support it (any column can be the label) but the framing of
the script is variant + service genre analysis only. See the related notebook
for a discussion of why the category-classification approach was attempted and
abandoned.


WHY MASKING MATTERS
-------------------
Rationale text contains several kinds of meta-information that leak class
identity into the feature space without revealing real rhetorical genre:

* Service names — the judge prompt labels each candidate as "(variant — Service)",
  so service tokens like "openai", "claude", "gemini", "ollama" appear inside
  rationales as meta-commentary, not as stylistic signature.
* Variant names — same problem. "expert_persona", "minimal", "judge",
  "native_rationale" are echoed by the judge in synthesis text.
* Source term — every rationale discusses "Digital Humanities", "digital",
  "humanities". These words alone aren't useful features but their CONTEXTS are
  (e.g., "borrowed from English" is a real Claude signal). Stripping the
  unigrams while keeping bigrams is a defensible compromise.
* Language name — "Burmese", "Cayuga", "Gothic" etc. appear in rationales
  describing each language. A classifier that learns language identity is not
  learning rhetorical style.
* Target translations — the translated term itself often appears verbatim
  inside the rationale that justifies it. These language-specific tokens are
  rare globally so min_df=5 mostly handles them; explicit stripping is
  available as a stricter ablation.


MASKING DESIGN: DELETION, NOT PLACEHOLDERS
------------------------------------------
Masking functions DELETE matched tokens rather than replacing them with a
placeholder string like [MASK] or [LANG]. Placeholders re-leak: TF-IDF's
default tokenizer strips brackets and produces "lang" / "mask" / "service" as
real features, simply renaming the leakage. Deletion (with whitespace
collapse) avoids this entirely.


TWO USAGE PATTERNS
------------------
1. SINGLE-PASS (the notebook's headline path):
   Apply mask_prompt_metadata() once to a rationale_raw column, store as
   rationale_clean, train a single classifier on the cleaned text. Reports the
   numbers used in the paper's main text.

2. ABLATION HARNESS (the supplementary table path):
   Use MaskingConfig + run_masking_ablation() to fit the same task under
   multiple progressively stricter masking presets (NONE, NAMES_ONLY, STANDARD,
   AGGRESSIVE), reporting how F1 and top features change as leakage is removed.
   Useful for a robustness footnote / supplementary table demonstrating that
   the genre signal survives aggressive masking.

Why logistic regression over a neural model:
  - Coefficients ARE the keyword list. No post-hoc explanation needed.
  - Fast enough to run interactively in a notebook.
  - Regularisation (C) controls overfitting without a separate validation split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score,
)


# ── Constants ──────────────────────────────────────────────────────────────

# English-language prompt variants. native_rationale is excluded from
# rationale-based classification because its rationales are in the target
# language; including them would teach the classifier script identity rather
# than rhetorical style.
ENGLISH_VARIANTS = ("minimal", "expert_persona", "judge")
SERVICES = ("claude", "openai", "gemini", "deepseek", "llama", "gemma", "qwen", "mistral")

# All variant and service tokens that appear verbatim in judge-prompt context.
# The judge prompt labels each candidate as "(variant — Service)", so models
# echo these strings back in their evaluation rationales. Includes
# native_rationale even though that variant itself is excluded from training,
# because judge prompts reference it.
_VARIANT_TOKENS = ("expert_persona", "native_rationale", "minimal", "judge")
_SERVICE_TOKENS = SERVICES

# Source-term unigrams. Stripping these is defensible because every rationale
# discusses the source concept; their bigram contexts ("borrowed digital",
# "digital electronic") often carry real signal so the bigram strip is opt-in.
SOURCE_TERM_UNIGRAMS = (
    "digital", "humanities", "humanity", "humanistic", "humanism", "dh",
)

# Compiled once at import time for efficiency. \b doesn't always behave as
# expected with Unicode but is fine for ASCII tokens (which all of these are).
_VARIANT_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _VARIANT_TOKENS) + r")\b",
    flags=re.IGNORECASE,
)
_SERVICE_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _SERVICE_TOKENS) + r")\b",
    flags=re.IGNORECASE,
)
_SOURCE_UNIGRAM_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in SOURCE_TERM_UNIGRAMS) + r")\b",
    flags=re.IGNORECASE,
)
_SOURCE_BIGRAM_RE = re.compile(
    r"\bdigital\s+humanit(?:y|ies|ie)?\b",
    flags=re.IGNORECASE,
)


# ── Single-purpose masking primitives ──────────────────────────────────────

def _collapse_whitespace(text: str) -> str:
    return re.sub(r" {2,}", " ", text).strip()


def mask_prompt_metadata(text: str) -> str:
    """Delete variant and service name tokens from rationale text.

    The judge prompt labels every candidate as "(variant — Service)", so models
    echo those strings back in their rationales. Deletion (rather than a
    placeholder) prevents the classifier from learning these metadata tokens
    OR a renamed proxy for them.
    """
    if not text or not isinstance(text, str):
        return text
    out = _VARIANT_RE.sub("", text)
    out = _SERVICE_RE.sub("", out)
    return _collapse_whitespace(out)


def mask_source_term(
    text: str,
    strip_bigrams: bool = False,
) -> str:
    """Delete source-term tokens (Digital Humanities and its components).

    By default strips only unigrams (digital, humanities, etc.) — the bigram
    "digital humanities" is preserved because its bigram context can carry
    real signal. Pass strip_bigrams=True for the stricter ablation setting.
    """
    if not text or not isinstance(text, str):
        return text
    out = text
    if strip_bigrams:
        out = _SOURCE_BIGRAM_RE.sub("", out)
    out = _SOURCE_UNIGRAM_RE.sub("", out)
    return _collapse_whitespace(out)


def mask_language_name(
    text: str,
    language_name: str,
    language_code: str = "",
) -> str:
    """Delete language name (and each long word within it) from text.

    Also deletes individual words in the language name that are longer than
    3 characters, so "Burmese" is caught even when the full name appears as
    "Burmese (Myanmar)". Common words like "language" / "languages" are
    excluded to avoid clobbering them.
    """
    if not text or not isinstance(text, str):
        return text
    if not language_name:
        return text
    out = re.sub(re.escape(language_name), "", text, flags=re.IGNORECASE)
    for word in language_name.split():
        if len(word) > 3 and word.lower() not in {"language", "languages"}:
            out = re.sub(
                r"\b" + re.escape(word) + r"\b", "", out, flags=re.IGNORECASE,
            )
    if language_code and len(language_code) >= 2:
        out = re.sub(
            r"\b" + re.escape(language_code) + r"\b", "", out,
            flags=re.IGNORECASE,
        )
    return _collapse_whitespace(out)


def mask_translated_term(text: str, translated_term: str) -> str:
    """Delete a per-row translated term and its component words (>3 chars)."""
    if not text or not isinstance(text, str) or not translated_term:
        return text
    out = re.sub(re.escape(translated_term), "", text, flags=re.IGNORECASE)
    for word in translated_term.split():
        if len(word) > 3:
            out = re.sub(
                r"\b" + re.escape(word) + r"\b", "", out, flags=re.IGNORECASE,
            )
    return _collapse_whitespace(out)


# ── Layered masking config (for ablation) ──────────────────────────────────

@dataclass
class MaskingConfig:
    """Layered masking choices for ablation experiments.

    Each flag toggles one masking primitive. The ablation harness runs the same
    classification task under several presets (NONE → AGGRESSIVE) and reports
    how performance and top features change as leakage is progressively
    removed.
    """
    mask_prompt_metadata: bool = True
    mask_source_unigrams: bool = True
    mask_source_bigrams: bool = False
    mask_language_names: bool = True
    mask_translated_terms: bool = False
    label: str = "STANDARD"


PRESET_NONE = MaskingConfig(
    mask_prompt_metadata=False, mask_source_unigrams=False,
    mask_source_bigrams=False, mask_language_names=False,
    mask_translated_terms=False, label="NONE",
)
PRESET_NAMES_ONLY = MaskingConfig(
    mask_prompt_metadata=True, mask_source_unigrams=False,
    mask_source_bigrams=False, mask_language_names=False,
    mask_translated_terms=False, label="NAMES_ONLY",
)
PRESET_STANDARD = MaskingConfig(label="STANDARD")  # all defaults
PRESET_AGGRESSIVE = MaskingConfig(
    mask_prompt_metadata=True, mask_source_unigrams=True,
    mask_source_bigrams=True, mask_language_names=True,
    mask_translated_terms=True, label="AGGRESSIVE",
)


def mask_rationale(
    text: str,
    language_name: str = "",
    language_code: str = "",
    translated_term: str = "",
    config: MaskingConfig = PRESET_STANDARD,
) -> str:
    """Apply a MaskingConfig to one rationale, calling primitives in order.

    Order matters: longer phrases mask before shorter ones, so the source-term
    bigram strip runs before the unigram strip.
    """
    if not text or not isinstance(text, str):
        return ""
    out = text
    if config.mask_prompt_metadata:
        out = mask_prompt_metadata(out)
    if config.mask_source_bigrams or config.mask_source_unigrams:
        out = mask_source_term(out, strip_bigrams=config.mask_source_bigrams)
    if config.mask_language_names:
        out = mask_language_name(out, language_name, language_code)
    if config.mask_translated_terms and translated_term:
        out = mask_translated_term(out, translated_term)
    return out


def apply_masking(
    df: pd.DataFrame,
    text_col: str,
    language_name_col: str = "language_name",
    language_code_col: str = "language_code",
    translated_term_col: Optional[str] = None,
    config: MaskingConfig = PRESET_STANDARD,
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """Apply mask_rationale row-by-row, returning a copy with new column.

    out_col defaults to f"{text_col}_masked_{config.label.lower()}".
    """
    if out_col is None:
        out_col = f"{text_col}_masked_{config.label.lower()}"
    df = df.copy()
    df[out_col] = [
        mask_rationale(
            row[text_col],
            language_name=row.get(language_name_col, "") or "",
            language_code=row.get(language_code_col, "") or "",
            translated_term=(row.get(translated_term_col, "") or "")
                            if translated_term_col else "",
            config=config,
        )
        for _, row in df.iterrows()
    ]
    return df


# ── TF-IDF and classification ──────────────────────────────────────────────

def build_tfidf(
    texts: List[str],
    extra_stopwords: Optional[Sequence[str]] = None,
    **kwargs,
) -> Tuple[object, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer on texts. Returns (sparse matrix, vectorizer).

    Defaults: word unigrams + bigrams, min_df=5, max_df=0.7, sublinear TF
    scaling, English stopwords plus any extras supplied. min_df=5 / max_df=0.7
    are stricter than older defaults because rationale corpora have a long
    tail of proper nouns (place names, language names, transliterated terms)
    that even careful masking misses; min_df=5 filters most of these without
    hurting rhetorical-vocabulary signal that appears in dozens of documents.

    Override any default via kwargs (e.g. min_df=3 to relax the threshold).
    """
    stop = set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stop |= set(extra_stopwords)

    params: dict = dict(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        sublinear_tf=True,
        max_features=15_000,
        stop_words=list(stop),
    )
    params.update(kwargs)
    vec = TfidfVectorizer(**params)
    X = vec.fit_transform(texts)
    return X, vec


def train_classifier(
    X,
    y: np.ndarray,
    C: float = 1.0,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Train logistic regression with stratified cross-validation.

    Returns (fitted_clf, cv_macro_f1_scores, out_of_fold_predictions).
    Out-of-fold predictions use the same folds as cv_scores, so the confusion
    matrix built from them is honest (never sees test labels during training).
    """
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state,
    )
    clf = LogisticRegression(
        C=C, max_iter=2000, random_state=random_state, solver="lbfgs",
    )
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    oof_preds = cross_val_predict(clf, X, y, cv=skf)
    clf.fit(X, y)
    return clf, cv_scores, oof_preds


def top_coefficients(
    clf: LogisticRegression,
    vectorizer: TfidfVectorizer,
    class_names: List[str],
    top_n: int = 20,
) -> Dict[str, List[Tuple[str, float]]]:
    """Top_n most positive features per class as (term, log-odds) pairs.

    For one-vs-rest logistic regression, a high positive coefficient means the
    term strongly predicts membership in that class over the union of all
    others. Note: top-feature RANKINGS are not stable across random seeds when
    multiple correlated features carry similar signal — for paper-quality
    feature lists, run with multiple seeds and keep features that appear in
    most runs (see stable_features() below).
    """
    feat = vectorizer.get_feature_names_out()
    results: Dict[str, List[Tuple[str, float]]] = {}
    for i, name in enumerate(class_names):
        coef = clf.coef_[i]
        idx = np.argsort(coef)[::-1][:top_n]
        results[name] = [(feat[j], float(coef[j])) for j in idx]
    return results


def stable_features(
    texts: List[str],
    y: np.ndarray,
    class_names: List[str],
    seeds: Sequence[int] = (5, 42, 100, 200, 500),
    top_n: int = 15,
    min_appearances: int = 4,
    tfidf_kwargs: Optional[dict] = None,
) -> Dict[str, List[Tuple[str, int]]]:
    """Find features stable across multiple random seeds.

    Runs the classifier `len(seeds)` times, each time taking the top_n
    features per class, and returns features that appear in at least
    `min_appearances` of those runs. Returns dict {class: [(term, count), ...]}
    sorted by appearance count descending.

    Use this for paper-quality feature tables — top-feature rankings are
    sensitive to seed when multiple correlated features carry similar signal,
    so a single run's top-15 may be misleading.
    """
    from collections import Counter
    tfidf_kwargs = tfidf_kwargs or {}
    appearances: Dict[str, Counter] = {c: Counter() for c in class_names}
    for seed in seeds:
        X, vec = build_tfidf(texts, **tfidf_kwargs)
        clf, _, _ = train_classifier(X, y, random_state=seed)
        coefs = top_coefficients(clf, vec, class_names, top_n=top_n)
        for cls, pairs in coefs.items():
            for term, _ in pairs:
                appearances[cls][term] += 1
    out: Dict[str, List[Tuple[str, int]]] = {}
    for cls, ctr in appearances.items():
        stable = [(t, c) for t, c in ctr.most_common() if c >= min_appearances]
        out[cls] = stable
    return out


def confusion_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> pd.DataFrame:
    """Tidy long-format confusion matrix suitable for an Altair heatmap.

    Each row has (true, predicted, n, pct) where pct is recall-normalised so
    the diagonal reads as per-class recall.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, true_lab in enumerate(labels):
        row_total = cm[i].sum()
        for j, pred_lab in enumerate(labels):
            n = int(cm[i, j])
            rows.append({
                "true": true_lab,
                "predicted": pred_lab,
                "n": n,
                "pct": round(n / row_total * 100, 1) if row_total > 0 else 0.0,
            })
    return pd.DataFrame(rows)


# ── Ablation harness ───────────────────────────────────────────────────────

def run_masking_ablation(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    class_order: List[str],
    presets: Sequence[MaskingConfig] = (
        PRESET_NONE, PRESET_NAMES_ONLY, PRESET_STANDARD, PRESET_AGGRESSIVE,
    ),
    language_name_col: str = "language_name",
    language_code_col: str = "language_code",
    translated_term_col: Optional[str] = None,
    top_n_features: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the same classification task under each masking preset.

    Reports macro-F1 and the top features per class for each preset so you
    can see how progressively masking leakage changes the classifier's
    performance. The flat returned DataFrame is suitable for a paper
    supplementary table.

    Returns a tidy DataFrame:
        masking, class, rank, term, weight, cv_f1_mean, cv_f1_std
    """
    rows: List[Dict] = []
    for cfg in presets:
        masked = apply_masking(
            df, text_col,
            language_name_col=language_name_col,
            language_code_col=language_code_col,
            translated_term_col=translated_term_col,
            config=cfg,
            out_col="_masked",
        )
        texts = masked["_masked"].tolist()
        y = masked[label_col].to_numpy()

        X, vec = build_tfidf(texts)
        clf, cv_scores, _ = train_classifier(X, y)
        f1_mean, f1_std = float(cv_scores.mean()), float(cv_scores.std())
        coefs = top_coefficients(clf, vec, class_order, top_n=top_n_features)

        if verbose:
            print(f"\n══ {cfg.label} | macro-F1 = {f1_mean:.3f} ± {f1_std:.3f} ══")
            for cls, pairs in coefs.items():
                top10 = ", ".join(f"{t}({w:+.2f})" for t, w in pairs[:10])
                print(f"  {cls:>16}: {top10}")

        for cls, pairs in coefs.items():
            for rank, (term, weight) in enumerate(pairs, 1):
                rows.append({
                    "masking": cfg.label,
                    "class": cls,
                    "rank": rank,
                    "term": term,
                    "weight": weight,
                    "cv_f1_mean": f1_mean,
                    "cv_f1_std": f1_std,
                })

    return pd.DataFrame(rows)
