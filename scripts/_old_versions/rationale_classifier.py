#!/usr/bin/env python3
"""
rationale_classifier.py
=======================
Interpretable text classifiers for data-driven keyword discovery.

Two tasks:
  1. Variant classification — predict which prompt variant (minimal, expert_persona,
     judge) produced a rationale. Top log-odds coefficients per class surface the
     rhetorical vocabulary that distinguishes each prompt style.

  2. Category classification — predict the disagreement category assigned by
     explore_disagreements_revised.py. Top positive coefficients for
     STRUCTURAL_ABSENCE are candidate replacements/extensions for the hand-curated
     ABSENCE_KEYWORDS list in that script.

Both use TF-IDF (word unigrams + bigrams) + logistic regression. Coefficients are
directly interpretable as log-odds ratios over the class complement — no black-box
importance scores.

Why logistic regression over a neural model:
  - Coefficients are the keyword list. No post-hoc explanation needed.
  - Fast enough to run interactively in a notebook.
  - Regularisation (C) controls overfitting without a separate validation split.

Data leakage note:
  The judge prompt supplies every variant name and service name as explicit labels
  in the prompt context (e.g. "expert_persona — OpenAI"). Models then repeat those
  labels in their evaluation text, so "expert_persona", "openai", etc. appear as
  surface tokens in judge rationales. Call mask_prompt_metadata() before building
  TF-IDF features to prevent the classifier from learning these metadata tokens
  instead of rhetorical style.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.multiclass import OneVsRestClassifier


# English-language prompt variants — native_rationale is excluded because its
# rationales are in the target language and would teach the classifier script
# identity rather than rhetorical style.
ENGLISH_VARIANTS = {"minimal", "expert_persona", "judge"}

LANG_PLACEHOLDER = "[LANG]"

# All variant and service tokens that appear verbatim in judge prompt context.
# The judge prompt labels each candidate as "(variant — Service)", so models
# echo these strings back in their evaluation rationales.
_VARIANT_TOKENS = ["expert_persona", "native_rationale", "minimal", "judge"]
_SERVICE_TOKENS = ["openai", "claude", "gemini", "ollama"]

# Compiled once at import time for efficiency
_VARIANT_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _VARIANT_TOKENS) + r")\b",
    flags=re.IGNORECASE,
)
_SERVICE_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _SERVICE_TOKENS) + r")\b",
    flags=re.IGNORECASE,
)


def mask_prompt_metadata(text: str) -> str:
    """Delete variant and service name tokens from text.

    The judge prompt labels every candidate as "(variant — Service)", so models
    echo those strings back in their rationales. Deleting (not replacing with a
    placeholder) prevents the classifier from learning these metadata tokens.

    Placeholders like [SERVICE] are avoided because TF-IDF's default tokenizer
    strips brackets and produces 'service' / 'variant' as real features,
    reintroducing the leakage in a different form.
    """
    if not text or not isinstance(text, str):
        return text
    result = _VARIANT_RE.sub("", text)
    result = _SERVICE_RE.sub("", result)
    # Collapse runs of whitespace left by deletions
    return re.sub(r"  +", " ", result).strip()


def mask_language_name(text: str, language_name: str, language_code: str = "") -> str:
    """Replace language name (and each long word within it) with [LANG].

    Also replaces individual words in the language name that are longer than
    3 characters, so "Burmese" is caught even when the full name appears as
    "Burmese (Myanmar)" with a parenthetical qualifier. Short function words
    ("of", "the") are skipped to avoid clobbering common English vocabulary.
    """
    if not text or not isinstance(text, str):
        return text
    result = re.sub(re.escape(language_name), LANG_PLACEHOLDER,
                    text, flags=re.IGNORECASE)
    for word in language_name.split():
        if len(word) > 3:
            result = re.sub(
                r"\b" + re.escape(word) + r"\b",
                LANG_PLACEHOLDER, result, flags=re.IGNORECASE,
            )
    if language_code:
        result = re.sub(
            r"\b" + re.escape(language_code) + r"\b",
            LANG_PLACEHOLDER, result, flags=re.IGNORECASE,
        )
    return result


def build_tfidf(texts: List[str], **kwargs) -> Tuple[object, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer on texts. Returns (sparse matrix, vectorizer).

    Defaults: word unigrams + bigrams, min_df=3, sublinear TF scaling,
    15 000 features. Override any default via kwargs.
    """
    params: dict = dict(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.75,
        sublinear_tf=True,
        max_features=15_000,
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
    Out-of-fold predictions use the same folds as cv_scores, so the
    confusion matrix built from them is honest (never sees test labels
    during training).
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # OneVsRestClassifier: trains one binary LR per class (this class vs all others),
    # so coefficients are true OVR log-odds — a positive value means the term genuinely
    # appears more in that class than in the rest combined. The old lbfgs multinomial
    # (joint softmax) could assign high positive weights to terms that never appear in
    # a class, because large *negative* weights for a competing class inflate a
    # coefficient via softmax arithmetic. multi_class and liblinear multiclass were
    # both removed in sklearn 1.7; OneVsRestClassifier is the explicit OVR replacement.
    clf = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=2000, random_state=random_state, solver="lbfgs"),
    )
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    oof_preds = cross_val_predict(clf, X, y, cv=skf)
    clf.fit(X, y)
    return clf, cv_scores, oof_preds


def top_coefficients(
    clf,
    vectorizer: TfidfVectorizer,
    class_names: List[str],
    top_n: int = 20,
) -> Dict[str, List[Tuple[str, float]]]:
    """Return the top_n most positive features per class as (term, log-odds) pairs.

    For a one-vs-rest logistic regression, a high positive coefficient means
    the term strongly predicts membership in that class over all others.

    Handles both plain LogisticRegression (clf.coef_[i]) and
    OneVsRestClassifier (clf.estimators_[i].coef_[0]), which dropped its
    coef_ shortcut in sklearn 1.7.
    """
    feat = vectorizer.get_feature_names_out()
    results: Dict[str, List[Tuple[str, float]]] = {}
    use_estimators = hasattr(clf, "estimators_") and not hasattr(clf, "coef_")
    for i, name in enumerate(class_names):
        coef = clf.estimators_[i].coef_[0] if use_estimators else clf.coef_[i]
        idx = np.argsort(coef)[::-1][:top_n]
        results[name] = [(feat[j], float(coef[j])) for j in idx]
    return results


def confusion_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> pd.DataFrame:
    """Return a tidy long-format confusion matrix suitable for an Altair heatmap.

    Each row has (true, predicted, n, pct) where pct is n / row_total * 100,
    i.e. recall-normalised so the diagonal reads as per-class recall.
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
