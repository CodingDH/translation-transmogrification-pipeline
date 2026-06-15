"""
Classify mixed-script artifacts in translation term columns.

Six patterns are handled:

1. Pattern A — romanization helper: "native_term (romanization)" or "native / romaji". The model appended an unrequested Latin transliteration alongside the real translation. Fix: strip the parenthetical (and any slash-suffix), keep the primary-script portion. It is possible that sometimes the model will return the translation in the parentheses, so this pattern also requires manual review.

2. Pattern B — character-level noise: scripts are mixed/interleaved mid-word with no structural delimiter. Fix: null the translation as it is not salvageable.

3. Pattern C — colon-separated prefix: "Digital Humanities : native_term" The model echoed the source term before a colon separator. The prefix contains no characters from the dominant (native) script, so it is safe to strip. Fix: discard everything up to and including the colon, keep the suffix.

4. Pattern D — space-separated Latin prefix: "Digital Humanities के दिशा कौशल" Same as Pattern C but without a colon. The English source term is prepended with only whitespace. Only applied when the first token is purely Latin script; non-Latin leading tokens are never stripped because they may be the actual translation, not a source prefix. Fix: strip the leading Latin tokens, keep from the first non-Latin token onward.

5. Pattern E — equals-sign separator: "Digital Humanities = Panagbalikas iti Digital". The model used an equation format to show source = translation. Unlike Patterns C and D, both sides may share the same script (e.g. Latin-script translations), so this check runs before the mixed-script gate. Safe to strip only when the prefix is entirely Latin. Fix: discard everything up to and including the `=`, keep the suffix.

6. Pattern F — slash-wrapped delimiter: "/Dkawng Thaukhnawng/" or "//term//". The model wrapped the entire translation in forward slashes as pseudo-delimiters (distinct from Pattern A's "native / romaji" separator). Fix: strip leading and trailing slashes, keep the inner content.

A translation is considered "mixed" when a secondary script passes either of two tiers:

1. Tier 1 — heavy mixing: the minority script contributes ≥ HIGH_MINORITY_UNIQUE
distinct characters. At this level the interference is too broad to be orthographic (e.g. "بومدنی humanities Digital", Myene, Beja).

2. Tier 2 — moderate mixing at high fraction: the minority script contributes ≥ MIN_MINORITY_UNIQUE distinct characters AND accounts for ≥ HIGH_FRACTION of all script chars. This catches cases like Korean transliteration in Sinhala text (3 Hangul chars at 21%) and Ojibwa gibberish (3 Greek chars at 33%) without flagging Chechen (4 Latin chars at only 16%, where the Latin letters are likely palochka substitutes or homoglyphs typed on a Cyrillic keyboard).

In both tiers a baseline MIXED_THRESHOLD (10%) still applies — a single stray character that represents < 10% of the string is always ignored.

False-positive examples that are correctly exempted:
  - Ossetian æ, Chuvash ă/ĕ: 1 unique Latin char in otherwise Cyrillic text
  - Sogdian academic transliteration: 2 unique Greek modifier chars
  - Chechen palochka substitution: 4 unique Latin chars but only 16% of the string

CJK codepoints and Japanese syllabaries are treated as one script family because
their co-occurrence is normal in Japanese text.

Pattern → quality flag mapping
------------------------------
curate_translation() returns one of four actions. Notebook 02 §1.9 routes
these actions to columns in quality_flags.csv as follows:

    Action        | quality_flags.csv column | Triggered by
    'stripped'    | has_romanization         | Patterns A, C, D, E
    'nulled'      | has_mixed_script         | Pattern B
    'placeholder' | has_placeholder_term     | is_placeholder_term()
    'unchanged'   | (no flag raised)         | —

Note: the flag is named has_romanization because Pattern A (romanization
parentheticals) is the most common case, but it fires for all four stripping
patterns including source-prefix stripping (C/D/E).

Importable API
--------------
    curate_translation(text) -> (cleaned_text | None, action)
        action: 'unchanged' | 'stripped' | 'nulled' | 'placeholder'

    curate_df(df, term_cols=None) -> (curated_df, summary_df)
        Applies curate_translation to every *_translated_term column.
        summary_df rows: service, unchanged, stripped, nulled, placeholder

    is_placeholder_term(text) -> bool
        True when text is a refusal or placeholder rather than a real translation
        (e.g. 'untranslatable', 'no direct translation', 'Note: ...').

    has_source_leakage(text, source_term) -> bool
        True when text contains the untranslated English source term verbatim or
        its uppercase initials as a standalone token (e.g. 'DH').

    is_repetition_loop(text) -> bool
        True when a single whitespace token appears ≥4× and represents ≥30% of
        all tokens (hallucination loop, e.g. EasyNMT Vietnamese 'bình' × 17).

    has_extreme_term_length(text, max_chars=100) -> bool
        True when the stripped text exceeds max_chars characters.  Catches both
        hallucination loops and LLM disclaimer text that escaped placeholder detection.

    has_unicode_escape(text) -> bool
        True when text contains a literal \\uXXXX escape sequence instead of the
        rendered Unicode character (e.g. Ollama Shor '\\u041d\\u0430...').

    script_mix_detail(text) -> dict
        Returns raw mixing metrics without applying exclusion thresholds.
        Keys: dominant_script, secondary (list of {script, count, fraction,
        unique_chars}), any_mixing (bool — any secondary chars present at all),
        excluded_mixing (bool — same gate as the 'nulled' path in curate_translation).
        Returns {} for non-string or empty input.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List, Optional, Tuple

import pandas as pd

# ── script detection ──────────────────────────────────────────────────────────

from scripts.utils import char_script


MIXED_THRESHOLD = 0.10       # secondary script must be ≥10% of script chars (baseline gate)
HIGH_MINORITY_UNIQUE = 5     # tier 1: ≥5 distinct minority chars → always mixed
MIN_MINORITY_UNIQUE = 3      # tier 2 lower bound on distinct minority chars
HIGH_FRACTION = 0.20         # tier 2: fraction threshold paired with MIN_MINORITY_UNIQUE

_PAREN_RE = re.compile(r"\s*\([^()]{1,150}\)")
_SLASH_RE = re.compile(r"\s*/\s*.+$")
_COLON_PREFIX_RE  = re.compile(r"^([^:]{1,100}):\s*(.+)$", re.DOTALL)
_EQUALS_PREFIX_RE = re.compile(r"^([^=]{1,100})\s*=+\s*(.+)$", re.DOTALL)
_SLASH_WRAP_RE    = re.compile(r"^/+(.+?)/+$", re.DOTALL)
_PLACEHOLDER_RE = re.compile(
    # note: prefix has no trailing \b because it ends with a non-word char
    r"note\s*:"
    r"|\b(?:"
    r"untranslatable|unable\s+to\s+translate|"
    r"cannot\s+(?:be\s+)?(?:\w+\s+)?translat(?:e|ed)|"
    r"not\s+possible\s+to\s+(?:\w+\s+){0,2}translat(?:e|ed|ing)|"
    r"translation\s+not\s+possible|"
    r"not\s+applicable|"
    r"no\s+(?:direct\s+|known\s+|established\s+|standard\s+|confirmed\s+|widely\s+recognized\s+|accepted\s+)?translation(?:\s+available)?|"
    r"isn'?t\s+(?:a\s+)?(?:direct|widely\s+recognized|confirmed|accepted|known|established|standard)\s+translation|"
    r"translation\s+not\s+available|not\s+translatable|no\s+(?:direct\s+)?equivalent|"
    r"fictional[\s_]translation|fictional[\s_]term|"
    r"translation\s+may\s+not\s+exist|(?:precise\s+)?translation\s+does\s+not\s+exist"
    r")\b",
    re.IGNORECASE,
)

_UNICODE_ESCAPE_RE = re.compile(r'\\u[0-9a-fA-F]{4}')

# Phrases inside rationales where the LLM explicitly admits no real translation
# exists. The signal pattern (observed e.g. for Mistral/Sangir): the model
# returns a placeholder in the `translated_term` column but tells us in the
# `translation_rationale` column that it doesn't actually know the translation.
# English-only — fluent_speaker rationales (in the target language) are out of
# scope and should be skipped by the caller.
_REFUSAL_RATIONALE_RE = re.compile(
    r"\b(?:"
    # "does/do (not) have a [adj] translation/equivalent/term/word"
    r"do(?:es)?\s+not\s+have\s+(?:a|an|any)\s+(?:\w+\s+){0,2}(?:translation|equivalent|term|word)|"
    r"do(?:es)?n'?t\s+have\s+(?:a|an|any)\s+(?:\w+\s+){0,2}(?:translation|equivalent|term|word)|"
    # "without a/an [adj] (translation|equivalent|term|word)"
    r"without\s+(?:a|an|any)\s+(?:\w+\s+){0,2}(?:translation|equivalent|term|word)|"
    # "lacks a/an [adj] (translation|equivalent|term|word)"
    r"lacks?\s+(?:a|an|any)\s+(?:\w+\s+){0,2}(?:translation|equivalent|term|word)|"
    # "no [adj] translation/equivalent/term/word"
    r"no\s+(?:direct|standard|standardi[sz]ed|specific|exact|widely\s+\w+|commonly\s+\w+|established|known|formal|official|equivalent|precise)\s+(?:translation|equivalent|term|word)|"
    # "there is no [adj] (translation|equivalent|term|word)"
    r"there\s+is\s+no\s+(?:direct|specific|standard|standardi[sz]ed|widely\s+\w+|established|formal|official|known|exact)\s+(?:translation|equivalent|term|word)|"
    # "I/I am/I'm not aware of"
    r"i(?:'m|\s+am)?\s+not\s+aware\s+of|"
    # "I do/don't know of (any|a)"
    r"i\s+(?:do\s+not|don'?t)\s+know\s+of\s+(?:any|a)|"
    # "(is|are) not (commonly|widely|typically|usually|directly) (translated|used)"
    r"(?:is|are)\s+not\s+(?:commonly|widely|typically|usually|directly)\s+(?:translated|used)|"
    # "(typically|usually|often|generally) (not translated|left untranslated|kept in english|borrowed|used in english/as is)"
    r"(?:typically|usually|often|generally)\s+(?:not\s+translated|left\s+untranslated|kept\s+in\s+english|borrowed\s+(?:from|directly)?|used\s+(?:in\s+english|as\s+is))|"
    # "cannot/can't (find|provide|offer) a (translation|equivalent|term)"
    r"can(?:not|'?t)\s+(?:find|provide|offer)\s+(?:a|an|any)\s+(?:\w+\s+)?(?:translation|equivalent|term)"
    r")\b",
    re.IGNORECASE,
)


# Phrases inside rationales where the LLM explicitly tells us it produced a
# transliteration rather than a translation — phonetic mapping of the source
# term's characters into the target script. Methodologically distinct from
# refusal (model attempted a "translation" but it's just letters re-spelled)
# and from legitimate loanwords (existing borrowed words used in the target).
# English-only — fluent_speaker rationales (target language) are out of scope.
_TRANSLITERATION_RATIONALE_RE = re.compile(
    r"\b(?:"
    # Core: transliterate / transliteration in any inflection
    r"transliterat(?:e|ed|ion|ions|ing)|"
    # "phonetic [transliteration|rendering|representation|approximation|adaptation|spelling]"
    r"phonetic(?:al(?:ly)?)?\s+(?:transliteration|rendering|representation|approximation|spelling|adapt(?:ed|ation)?)|"
    # "rendered/spelled/written phonetically"
    r"(?:rendered|spelled|written)\s+phonetic(?:al(?:ly)?)?"
    r")\b",
    re.IGNORECASE,
)


# Phrases inside rationales where the LLM tells us it produced a deliberate
# placeholder term rather than a translation — e.g. repeated words used as a
# stand-in for a concept the model could not translate. This is distinct from
# is_placeholder_term (which flags placeholder TERMS like "Translation not
# available") and from is_refusal_rationale (which catches explicit "does not
# have a translation" admissions); this flag fires specifically on the word
# "placeholder" or near-synonyms appearing in the rationale text.
# English-only — fluent_speaker rationales (target language) are out of scope.
_PLACEHOLDER_RATIONALE_RE = re.compile(
    r"\b(?:"
    # Core: the word "placeholder" itself, in any context — highly specific
    r"placeholder|"
    # "stand-in for X" or "stands in for X"
    r"stand[\-\s]?in\s+for|"
    r"stands\s+in\s+for|"
    # "stopgap" — uncommon, clearly signals placeholder behaviour
    r"stopgap|"
    # "filler term/word/phrase"
    r"filler\s+(?:term|word|phrase)|"
    # "serves as a (placeholder|stand-in|substitute|stopgap)"
    r"serves?\s+as\s+(?:a\s+|an\s+)?(?:placeholder|stand[\-\s]?in|substitute|stopgap)"
    r")\b",
    re.IGNORECASE,
)


def has_unexpected_rationale_language(rationale: str, min_letters: int = 20,
                                       latin_threshold: float = 0.50) -> bool:
    """
    Return True when a rationale that is *expected* to be in English contains
    substantially more non-Latin letters than Latin ones — a signal that the
    model is writing the rationale in a different language than its prompt
    expected (e.g., Qwen generating a Tajik-target rationale entirely in
    Chinese, or DeepSeek generating an Avar-target rationale in Russian).

    Only meaningful for non-fluent_speaker variants (minimal, github_searcher,
    judge), where the rationale convention is English. Callers should skip
    fluent_speaker, which legitimately uses target-language rationales.

    Note: this will also fire (correctly) when models write Chinese rationales
    for Chinese-target translations in non-fluent_speaker variants — which is
    a real code-switching signal worth surfacing, not a false positive.

    Parameters
    ----------
    rationale : str
        The rationale text to check.
    min_letters : int
        Minimum letter-character count required before assessing (rationales
        shorter than this are not flagged — too little signal).
    latin_threshold : float
        Latin-letter fraction below which the rationale is flagged.

    Returns
    -------
    bool
        True if the rationale is majority non-Latin script.
    """
    if not isinstance(rationale, str) or not rationale.strip():
        return False
    letters = [c for c in rationale if c.isalpha()]
    if len(letters) < min_letters:
        return False
    n_latin = 0
    for c in letters:
        try:
            name = unicodedata.name(c)
        except ValueError:
            continue
        if name.startswith('LATIN'):
            n_latin += 1
    return (n_latin / len(letters)) < latin_threshold


def is_language_name_term(term: str, language_name: str) -> bool:
    """
    Return True when the translated term contains the target language's own
    name as a substantive component — catching the pass-through failure mode
    where the model returns the language name (e.g. 'Mbere', 'Tasawaq') or
    embeds it in the output (e.g. 'Warlpiri-jarra Digital Humanities',
    'Arawak Digital Humanities') instead of producing a translation.

    Distinct from `has_source_term` (which detects English source-term leakage
    like 'Digital Humanities' or 'DH'); this flag detects *target*-language-name
    leakage.

    Parameters
    ----------
    term : str
        The translated term to check.
    language_name : str
        The English name of the target language (from metadata).

    Returns
    -------
    bool
        True if the language name appears as a whole-word component of the term.
    """
    if not isinstance(term, str) or not term.strip(): return False
    if not isinstance(language_name, str) or not language_name.strip(): return False
    # Normalize: lowercase, replace punctuation with space, collapse whitespace
    def _norm(s):
        s = re.sub(r'[^\w\s]', ' ', s, flags=re.UNICODE)
        return re.sub(r'\s+', ' ', s).strip().lower()
    nterm = _norm(term)
    nlang = _norm(language_name)
    if not nterm or not nlang: return False
    if nterm == nlang: return True
    # Whole-word substring match
    return bool(re.search(r'\b' + re.escape(nlang) + r'\b', nterm))


def is_placeholder_rationale(rationale: str) -> bool:
    """
    Return True when a rationale explicitly admits that the produced term is a
    placeholder or stand-in rather than a translation. Methodologically distinct
    from is_placeholder_term (which detects placeholder *terms* like "Translation
    not available") and from is_refusal_rationale (which catches "does not have
    a translation" admissions); this flag fires on the word "placeholder" itself
    or near-synonyms appearing inside the rationale text.

    English-only — fluent_speaker rationales are in the target language and
    callers should skip that variant.

    Parameters
    ----------
    rationale : str
        The rationale text to check.

    Returns
    -------
    bool
        True if the rationale describes the term as a placeholder/stand-in.
    """
    if not isinstance(rationale, str) or not rationale.strip():
        return False
    return bool(_PLACEHOLDER_RATIONALE_RE.search(rationale))


def is_transliteration_rationale(rationale: str) -> bool:
    """
    Return True when a rationale describes the produced term as a
    transliteration (phonetic mapping of source-term characters into the target
    script) rather than a real translation. Methodologically distinct from
    `is_refusal_rationale` (the model explicitly admits no translation exists)
    and from legitimate loanwords (existing borrowed words native to the
    target language).

    English-only — fluent_speaker rationales are in the target language and
    callers should skip that variant.

    Parameters
    ----------
    rationale : str
        The rationale text to check.

    Returns
    -------
    bool
        True if the rationale describes a transliteration approach.
    """
    if not isinstance(rationale, str) or not rationale.strip():
        return False
    return bool(_TRANSLITERATION_RATIONALE_RE.search(rationale))


def is_refusal_rationale(rationale: str) -> bool:
    """
    Return True when a rationale contains language explicitly admitting that no
    real translation exists (e.g., "does not have a standardized translation",
    "no direct equivalent", "I am not aware of"). Signals that the model is
    telling us in the rationale column that the term it returned in the
    translated_term column is a placeholder, not a real translation.

    English-only — fluent_speaker rationales are in the target language and
    callers should skip that variant.

    Parameters
    ----------
    rationale : str
        The rationale text to check.

    Returns
    -------
    bool
        True if the rationale contains an explicit refusal phrase.
    """
    if not isinstance(rationale, str) or not rationale.strip():
        return False
    return bool(_REFUSAL_RATIONALE_RE.search(rationale))


def is_placeholder_term(text: str) -> bool:
    """
    Return True if text is a refusal/placeholder rather than a real translation. Uses the _PLACEHOLDER_RE regex to catch common refusal patterns such as "untranslatable", "no direct translation", and "Note: ...".
    
    Parameters
    ----------
    text : str
        The text to check.

    Returns
    -------
    bool
        True if the text is a refusal/placeholder, False otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(_PLACEHOLDER_RE.search(text))


def is_repetition_loop(text: str) -> bool:
    """
    Return True when a single whitespace token dominates the string.

    Fires when the most-frequent token appears ≥4 times AND accounts for ≥30%
    of all tokens.  Catches EasyNMT/Gemini hallucination loops such as
    'bình bình bình ... bình' (Vietnamese) or 'kàlā kàlā kàlā ...' (Ngambay).
    The 30% fraction guard avoids false-positives from high-frequency function
    words in short strings (e.g. Arabic definite article or Bantu class prefixes).
    
    Parameters
    ----------
    text : str
        The text to check.

    Returns
    -------
    bool
        True if the text is a repetition loop, False otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    tokens = text.split()
    if len(tokens) < 4:
        return False
    _, freq = Counter(tokens).most_common(1)[0]
    return freq >= 4 and freq / len(tokens) >= 0.30


def has_extreme_term_length(text: str, max_chars: int = 100) -> bool:
    """
    Return True when a translation term exceeds max_chars characters.

    Flags two distinct failure modes: (a) hallucination loops whose tokens are
    long enough that character length triggers before word-count does (e.g.
    Lozi EasyNMT 109-char repeated phrase), and (b) LLM disclaimer text that
    slipped past placeholder detection (e.g. 312-char 'As of my knowledge
    cutoff...' responses).
    
    Parameters
    ----------
    text : str
        The text to check.
    max_chars : int, optional
        The maximum number of characters allowed, by default 100

    Returns
    -------
    bool
        True if the text exceeds max_chars characters, False otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return len(text.strip()) > max_chars


def has_short_translation(text: str, min_codepoints: int = 4) -> bool:
    """
    Return True when a translation has fewer than min_codepoints Unicode codepoints.

    Single- or two-codepoint translations produce excessive false positives in
    GitHub search regardless of script. Uses codepoint count (not byte length or
    UTF-16 code-unit length) so supplementary-plane scripts like Miao/Pollard or
    Linear B are measured correctly.

    Parameters
    ----------
    text : str
        The text to check.
    min_codepoints : int, optional
        Minimum acceptable codepoint count, by default 4.

    Returns
    -------
    bool
        True if the text has fewer than min_codepoints codepoints, False otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return len(text.strip()) > 0 and len(list(text.strip())) < min_codepoints


def has_unicode_escape(text: str) -> bool:
    """
    Return True when text contains a literal \\uXXXX escape sequence.

    Indicates the model emitted raw JSON/Python Unicode escapes instead of
    rendering the actual characters (e.g. Shor/Ollama: '\\u041d\\u0430...').
    
    Parameters
    ----------
    text : str
        The text to check.

    Returns
    -------
    bool
        True if the text contains a literal \\uXXXX escape sequence, False otherwise.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(_UNICODE_ESCAPE_RE.search(text))


def _script_profile(text: str) -> dict[str, int]:
    """
    Return {script: char_count}, ignoring punctuation/digits/spaces.
    
    Punctuation, symbols, separators, and digits are ignored as they do not reliably indicate script mixing. CJK codepoints and Japanese syllabaries are treated as one script family because their co-occurrence is normal in Japanese text.
    
    Parameters
    ----------
    text : str
        The text to analyze.

    Returns
    -------
    dict[str, int]
        A dictionary mapping scripts to character counts, ignoring punctuation/digits/spaces.
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    counts: Counter = Counter()
    for ch in text:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if cat[0] in ("Z", "P", "S", "C") or cat == "Nd":
            continue
        s = char_script(cp)
        if s == "Other":
            continue
        # CJK kanji + Japanese syllabaries co-occur normally — treat as one family
        s = "CJK/Japanese" if s in ("CJK", "Japanese") else s
        counts[s] += 1
    return dict(counts)


def _is_mixed(
    profile: dict[str, int],
    text: str = "",
    threshold: float = MIXED_THRESHOLD,
    high_minority_unique: int = HIGH_MINORITY_UNIQUE,
    min_minority_unique: int = MIN_MINORITY_UNIQUE,
    high_fraction: float = HIGH_FRACTION,
) -> bool:
    """
    Two-tier test for genuine script mixing (see module docstring for rationale).

    Tier 1 — unique ≥ high_minority_unique: unambiguous mixing regardless of fraction.
    Tier 2 — unique ≥ min_minority_unique AND fraction ≥ high_fraction: moderate mixing confirmed by both breadth and prevalence.

    A minority script that passes neither tier is treated as orthographic extension.
    """
    if len(profile) < 2:
        return False
    total = sum(profile.values())
    if total == 0:
        return False
    dominant = max(profile, key=profile.get)
    for script, count in profile.items():
        if script == dominant:
            continue
        frac = count / total
        if frac < threshold:
            continue
        if text:
            unique = {
                ch for ch in text
                if (
                    char_script(ord(ch)) in ("CJK", "Japanese")
                    if script == "CJK/Japanese"
                    else char_script(ord(ch)) == script
                )
            }
            n = len(unique)
            if n >= high_minority_unique:        # tier 1
                return True
            if n >= min_minority_unique and frac >= high_fraction:  # tier 2
                return True
        else:
            return True
    return False


def script_mix_detail(text: str) -> dict:
    """
    Return raw script-mixing metrics without applying exclusion thresholds.

    Useful for downstream analysis where mixing degree matters, not just
    whether it crosses the exclusion gate.

    Returns
    -------
    dict with keys:
        dominant_script : str — plurality script (most characters)
        secondary       : list[dict] — one entry per non-dominant script,
                          each with {script, count, fraction, unique_chars},
                          sorted by fraction descending
        any_mixing      : bool — True if any secondary-script chars are present,
                          regardless of threshold (superset of excluded_mixing)
        excluded_mixing : bool — True if mixing crosses the two-tier threshold
                          (identical gate to the 'nulled' path in curate_translation)

    Returns {} for non-string or empty/whitespace input.
    """
    if not isinstance(text, str) or not text.strip():
        return {}

    profile = _script_profile(text)
    if not profile:
        return {}

    total = sum(profile.values())
    dominant = max(profile, key=profile.get)

    secondary = []
    for script, count in profile.items():
        if script == dominant:
            continue
        fraction = count / total
        unique = {
            ch for ch in text
            if (
                char_script(ord(ch)) in ("CJK", "Japanese")
                if script == "CJK/Japanese"
                else char_script(ord(ch)) == script
            )
        }
        secondary.append({
            "script": script,
            "count": count,
            "fraction": round(fraction, 4),
            "unique_chars": len(unique),
        })

    return {
        "dominant_script": dominant,
        "secondary": sorted(secondary, key=lambda x: -x["fraction"]),
        "any_mixing": len(secondary) > 0,
        "excluded_mixing": _is_mixed(profile, text),
    }


# ── public API ────────────────────────────────────────────────────────────────

def has_source_leakage(text: str, source_term: str) -> bool:
    """
    Return True if text contains the untranslated English source term.

    Checks for:
      - the full term verbatim (case-insensitive): "Digital Humanities"
      - its uppercase initials as a standalone token: "DH"

    Intended for non-English translations only; callers should skip
    language_code == 'en'.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if re.search(re.escape(source_term), text, re.IGNORECASE):
        return True
    initials = "".join(w[0].upper() for w in source_term.split() if w)
    if len(initials) >= 2 and re.search(r"\b" + re.escape(initials) + r"\b", text):
        return True
    return False


def curate_translation(text: str) -> Tuple[Optional[str], str]:
    """
    Classify and optionally repair one translation string.

    Returns
    -------
    (result, action) where action is one of:
        'unchanged'   — no issues; returned as-is
        'stripped'    — parenthetical romanization removed; primary script kept
        'nulled'      — interleaved noise; not salvageable → result is None
        'placeholder' — refusal/placeholder text, not a real translation → result is None
    """
    if not isinstance(text, str) or not text.strip():
        return text, "unchanged"

    if is_placeholder_term(text):
        return None, "placeholder"

    # Pattern E — equals-sign separator: "Digital Humanities = Panagbalikas iti Digital"
    # Must run before the mixed-script gate because both sides may share the same script,
    # so _is_mixed returns False and the string would otherwise pass through unchanged.
    # Safe to strip only when the prefix is entirely Latin (i.e. an echoed English source term).
    m = _EQUALS_PREFIX_RE.match(text)
    if m:
        pre, post = m.group(1).strip(), m.group(2).strip()
        pre_profile = _script_profile(pre)
        if post and pre_profile and set(pre_profile.keys()) == {"Latin"}:
            if not _is_mixed(_script_profile(post), post):
                return post, "stripped"

    # Pattern F — slash-wrapped delimiter: "/Dkawng Thaukhnawng/"
    # The model wrapped the entire translation in forward slashes as pseudo-delimiters.
    # Distinct from Pattern A ("native / romaji") which uses a slash as an inline separator.
    # Runs before the mixed-script gate so purely Latin-script terms are also cleaned.
    m = _SLASH_WRAP_RE.match(text)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner, "stripped"

    profile = _script_profile(text)
    if not _is_mixed(profile, text):
        return text, "unchanged"

    # Try stripping parentheticals, then slash suffix
    candidates = []
    stripped_parens = _PAREN_RE.sub("", text).strip()
    if stripped_parens:
        candidates.append(stripped_parens)
        stripped_slash = _SLASH_RE.sub("", stripped_parens).strip()
        if stripped_slash and stripped_slash != stripped_parens:
            candidates.append(stripped_slash)

    dominant = max(profile, key=profile.get)

    # Pattern C — colon-separated prefix: "Digital Humanities : native_term"
    # Safe to strip only if the prefix contains no characters from the dominant script.
    m = _COLON_PREFIX_RE.match(text)
    if m:
        pre, post = m.group(1).strip(), m.group(2).strip()
        if post and dominant not in _script_profile(pre):
            candidates.append(post)

    # Pattern D — space-separated Latin prefix: "Digital Humanities के दिशा कौशल"
    # Only applied when the first token is purely Latin — this is the English
    # source-term prefix case. Non-Latin leading tokens (Bengali, Devanagari, etc.)
    # are never stripped: they may be the actual translation, not a source prefix.
    tokens = text.split()
    if tokens:
        first_tok_profile = _script_profile(tokens[0])
        if len(first_tok_profile) == 1 and next(iter(first_tok_profile)) == "Latin":
            prefix_script = "Latin"
            prefix_end = 0
            for tok in tokens:
                tok_p = _script_profile(tok)
                if not tok_p or set(tok_p.keys()) - {prefix_script}:
                    break
                prefix_end += 1
            if 0 < prefix_end < len(tokens):
                candidates.append(" ".join(tokens[prefix_end:]))

    for candidate in candidates:
        if candidate and not _is_mixed(_script_profile(candidate), candidate):
            return candidate, "stripped"

    return None, "nulled"


def curate_df(
    df: pd.DataFrame,
    term_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply curate_translation to every *_translated_term column in df.

    Parameters
    ----------
    df        : DataFrame with translation columns (modified copy is returned)
    term_cols : explicit list of columns to clean; defaults to all columns
                ending in '_translated_term'

    Returns
    -------
    curated_df  : copy of df with cleaned values (nulled → NaN)
    summary_df  : one row per service with counts: unchanged, stripped, nulled
    """
    out = df.copy()
    if term_cols is None:
        term_cols = [c for c in df.columns if c.endswith("_translated_term")]

    summary_rows = []
    for col in term_cols:
        service = col.replace("_translated_term", "")
        counts = {"unchanged": 0, "stripped": 0, "nulled": 0, "placeholder": 0}
        curated_vals = []
        for val in out[col]:
            result, action = curate_translation(val)
            counts[action] += 1
            curated_vals.append(result)
        out[col] = curated_vals
        summary_rows.append({"service": service, **counts})

    summary_df = pd.DataFrame(summary_rows)
    return out, summary_df
