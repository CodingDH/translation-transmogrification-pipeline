"""
Standalone utility functions for the translation pipeline.

Some of these functions are extracted from the parent CodingDH utils.py, so this repo is self-contained and can be installed independently.

Script detection
----------------
_char_script(cp) maps a Unicode code point to a script family name (e.g. 'Latin',
'Cyrillic', 'Arabic', 'CJK', 'Japanese', 'New_Tai_Lue', …).

When the `regex` package is available (recommended), a single compiled alternation of Unicode Script property patterns covers 55+ scripts and uses functools.lru_cache so each unique codepoint is matched at most once per process.  If `regex` is not installed the function falls back to a hand-coded range table covering the ~30 most common scripts.

detect_dominant_script(text) calls _char_script on every letter/mark character in a string and returns the plurality script, ignoring punctuation, digits, and spaces.
"""

from __future__ import annotations

import functools
import os
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rich.console import Console
import apikey

console = Console()


def read_csv_file(file_path: str, error_bad_lines: bool = True) -> pd.DataFrame:
	"""
	Read a CSV file with UTF-8 encoding, falling back to latin-1 on failure.

	Parameters
	----------
	file_path : str
	error_bad_lines : bool
		Passed to pandas; set False to skip malformed rows.

	Returns
	-------
	pd.DataFrame
	"""
	bad_lines = 'skip' if not error_bad_lines else 'error'
	# converters preserves 'nan' (Min Nan Chinese, ISO 639-3) as the string "nan" rather than letting pandas promote it to float NaN via its default NA list.
	_conv = {'language_code': str}
	try:
		return pd.read_csv(file_path, encoding='utf-8', on_bad_lines=bad_lines, converters=_conv)
	except UnicodeDecodeError:
		return pd.read_csv(file_path, encoding='latin-1', on_bad_lines=bad_lines, converters=_conv)
	except pd.errors.ParserError:
		# Fall back to the Python engine, which is more lenient with malformed rows (e.g. column-count mismatches from schema migrations).
		return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python', converters=_conv)


def log_error_to_file(
	error_file_path: str,
	additional_data: dict,
	status_code: int,
	error_url: str,
	error_message: str = "",
) -> None:
	"""
	Append a structured error record to a CSV error log.

	Parameters
	----------
	error_file_path : str
	additional_data : dict
		Extra columns to record (e.g. term_source, language_code).
	status_code : int
	error_url : str
		A description of the failing call.
	error_message : str
		The full error message from the service or exception.
	"""
	error_df = pd.DataFrame([{
		"error_date": datetime.now().strftime("%Y-%m-%d"),
		"error_url": error_url,
		"status_code": status_code,
		"error_message": error_message,
	}])
	error_df = pd.concat([error_df, pd.DataFrame([additional_data])], axis=1)

	if os.path.exists(error_file_path):
		existing = pd.read_csv(error_file_path, nrows=0)
		missing_cols = [c for c in error_df.columns if c not in existing.columns]
		if missing_cols:
			# Schema migration: backfill any new columns so existing rows stay aligned.
			existing_full = read_csv_file(error_file_path, error_bad_lines=False)
			for col in missing_cols:
				existing_full[col] = ''
			existing_full.to_csv(error_file_path, index=False)
			# Re-read header after migration so reindex below uses the updated schema.
			existing = pd.read_csv(error_file_path, nrows=0)
		# Reorder new row to match the file's column order exactly — mismatches cause values to land in the wrong columns when appending without a header.
		file_cols = existing.columns.tolist()
		for c in file_cols:
			if c not in error_df.columns:
				error_df[c] = None
		error_df = error_df.reindex(columns=file_cols)
		error_df.to_csv(error_file_path, mode='a', header=False, index=False)
	else:
		os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
		error_df.to_csv(error_file_path, index=False)


def clean_write_error_file(
	error_file_path: str,
	drop_fields: Optional[List[str]] = None,
) -> None:
	"""
	Deduplicate an error log file in place.

	Parameters
	----------
	error_file_path : str
	drop_fields : list of str, optional
		Column subset used for deduplication.
	"""
	if os.path.exists(error_file_path):
		error_df = read_csv_file(error_file_path, error_bad_lines=False)
		# Filter drop_fields to columns that actually exist so callers can include "variant" in the list without KeyErrors on legacy CSVs that lack the column.
		_dedup_fields = [f for f in (drop_fields or []) if f in error_df.columns] or None
		if 'error_date' in error_df.columns:
			error_df['error_date'] = pd.to_datetime(error_df['error_date'])
			error_df = error_df.sort_values('error_date').drop_duplicates(
				subset=_dedup_fields, keep='last'
			)
		elif _dedup_fields:
			error_df = error_df.drop_duplicates(subset=_dedup_fields, keep='last')
		# Promote permanent failures: drop any transient (500/408) entry for a (term, language[, variant]) tuple that already has a deterministic (400/404) entry. This prevents a stale 500 from hiding a 400 in mark_errored_terms.
		_TRANSIENT = {500, 408}
		if 'status_code' in error_df.columns:
			_has_variant = 'variant' in error_df.columns
			_perm = error_df[~error_df['status_code'].isin(_TRANSIENT)]
			if _has_variant:
				permanent_pairs = set(
					zip(
						_perm['term_source'].astype(str),
						_perm['language_code'].astype(str),
						_perm['variant'].astype(str),
					)
				)
				if permanent_pairs:
					redundant = (
						error_df['status_code'].isin(_TRANSIENT) &
						error_df.apply(
							lambda r: (str(r['term_source']), str(r['language_code']), str(r['variant'])) in permanent_pairs,
							axis=1,
						)
					)
					error_df = error_df[~redundant]
			else:
				permanent_pairs = set(
					zip(
						_perm['term_source'].astype(str),
						_perm['language_code'].astype(str),
					)
				)
				if permanent_pairs:
					redundant = (
						error_df['status_code'].isin(_TRANSIENT) &
						error_df.apply(
							lambda r: (str(r['term_source']), str(r['language_code'])) in permanent_pairs,
							axis=1,
						)
					)
					error_df = error_df[~redundant]
		error_df.to_csv(error_file_path, index=False)
	else:
		console.print('No error file to clean', style='bold blue')


def set_data_directory_path(path: str) -> None:
	"""
	Sets data directory path.

	:param path: Path to data directory
	"""
	apikey.save("TRANSLATION_TRANSMOGRIFICATION_DATA_DIRECTORY", path)
	console.print(f'Translation pipeline data directory path set to {path}', style='bold blue')

def get_data_directory_path() -> str:
	"""
	Gets data directory path.

	:return: Data directory path
	"""
	console.print('Retrieving translation pipeline data directory path...', style='bold blue')
	return apikey.load("TRANSLATION_TRANSMOGRIFICATION_DATA_DIRECTORY")


_LANG_FAMILY_CACHE: Optional[Dict[str, str]] = None


def get_language_family(code: str) -> str:
	"""Return the top-level family name for a language code, reading from the comprehensive language codes CSV. Falls back to 'Other' when unknown.

	Uses `family_name_reconciled` when present (the Glottolog-cross-validated column produced by generate_language_codes.py from the reconciliation table reviewed in notebook 01), falling back to `family_name` (ISO 639-5 based) when reconciled is unavailable. See docs/exclusion_strategy.md and notebook 01 §1.3 for the reconciliation methodology.
	"""
	global _LANG_FAMILY_CACHE
	if _LANG_FAMILY_CACHE is None:
		data_dir = get_data_directory_path()
		csv_path = os.path.join(data_dir, 'metadata_files', 'language_codes_comprehensive.csv')
		if os.path.exists(csv_path):
			# converters preserves 'nan' (Min Nan Chinese, ISO 639-3) as a string
			df = pd.read_csv(csv_path, converters={'language_code': str})
			# Prefer the reconciled column; fall back to family_name where reconciled is missing
			fam_col = (
				df['family_name_reconciled'].fillna(df['family_name'])
				if 'family_name_reconciled' in df.columns
				else df['family_name']
			)
			_LANG_FAMILY_CACHE = dict(
				zip(df['language_code'],
					fam_col.fillna('Other').astype(str))
			)
		else:
			_LANG_FAMILY_CACHE = {}
	return _LANG_FAMILY_CACHE.get(code, 'Other')


# ── Exclusion tiers ───────────────────────────────────────────────────────────
#
# Two sources feed filter_for_analysis():
#
# 1. quality_flags.csv (automated)
#    Produced by notebook 02.  Flags with an unambiguous interpretation are
#    applied automatically — you don't need to Xall every repetition loop.
#
#    likely_error (nulled in 'quality' and 'search_ready'):
#      has_repetition_loop, has_mixed_script, has_placeholder_term,
#      has_unicode_escape, has_extreme_term_length
#
#    likely_search_complication (nulled in 'search_ready' only):
#      has_romanization, has_source_term (non-English), has_short_translation
#
#    NOT auto-excluded (analytical signal, not errors):
#      has_script_disagreement, has_any_mixing, has_missing_rationale
#
# 2. manual_exclusions.csv (human review via review_explorer.html)
#    Supplements and overrides the automated flags for edge cases.
#    Xall  → likely_error               (exclude_translation = True)
#    Xsrch → likely_search_complication (per-variant boolean flags only)
#
# Analysis modes
# --------------
#   'full'         – no filtering  (error rates, coverage stats)
#   'quality'      – null likely_error cells  (model quality comparison)
#   'search_ready' – null both tiers  (building the translation resource)

_SVC_TO_PREFIX = {
    "Claude":           "claude",
    "OpenAI":           "openai",
    "Gemini":           "gemini",
    "DeepSeek":         "deepseek",
    "Llama":            "llama",
    "Gemma":            "gemma",
    "Qwen":             "qwen",
    "Mistral":          "mistral",
    "Google Translate": "gt",
    "EasyNMT":          "enmt",
    "Lingvanex":        "lingvanex",
    "Wikipedia":        "wikipedia",
}

_VARIANT_COL_TO_NAME = {
    "exclude_term_minimal":          "minimal",
    "exclude_term_fluent_speaker":   "fluent_speaker",
    "exclude_term_github_searcher":  "github_searcher",
    "exclude_term_judge":            "judge",
}


def load_exclusion_tiers(term: str, data_dir: str) -> pd.DataFrame:
	"""Return a DataFrame with columns (language_code, service, tier) where
	tier is 'likely_error', 'likely_search_complication', or 'keep'.

	Reads manual_exclusions.csv written by the review tool; returns an empty
	DataFrame if the file does not exist yet.
	"""
	path = os.path.join(
		data_dir, "translated_terms",
		term.lower().replace(" ", "_"),
		"evaluation", "manual_exclusions.csv",
	)
	if not os.path.exists(path):
		return pd.DataFrame(columns=["language_code", "service", "tier"])

	ex = pd.read_csv(path, converters={"language_code": str})
	variant_cols = list(_VARIANT_COL_TO_NAME.keys())

	rows = []
	for _, r in ex.iterrows():
		if r.get("exclude_translation") is True or r.get("exclude_translation") == True:
			tier = "likely_error"
		elif any(r.get(c) for c in variant_cols):
			tier = "likely_search_complication"
		else:
			tier = "keep"
		rows.append({
			"language_code": r["language_code"],
			"service":       r["service"],
			"tier":          tier,
			# preserve per-variant detail for search_ready nulling
			**{_VARIANT_COL_TO_NAME[c]: bool(r.get(c, False)) for c in variant_cols},
		})

	return pd.DataFrame(rows)


def filter_for_analysis(df: pd.DataFrame, term: str, data_dir: str, mode: str) -> pd.DataFrame:
	"""Return a copy of df with excluded cells nulled according to mode.

	Works on wide-format DataFrames (one row per language × prompt_variant)
	as produced by load_all_variants().  Translation columns are nulled in
	place; no rows are dropped, so error-rate denominators stay correct.

	Parameters
	----------
	df       : the wide-format DataFrame from all_dfs[term]
	term     : source term string, e.g. "Digital Humanities"
	data_dir : DATA_DIR from the notebook
	mode     : 'full' | 'quality' | 'search_ready'
	"""
	if mode == "full":
		return df.copy()

	tiers = load_exclusion_tiers(term, data_dir)
	if tiers.empty:
		return df.copy()

	out = df.copy()

	for _, r in tiers.iterrows():
		lc      = r["language_code"]
		prefix  = _SVC_TO_PREFIX.get(r["service"])
		if prefix is None:
			continue
		col = f"{prefix}_translated_term"
		if col not in out.columns:
			continue

		lang_mask = out["language_code"] == lc

		if r["tier"] == "likely_error":
			# null this service for ALL variants of this language
			out.loc[lang_mask, col] = None

		elif r["tier"] == "likely_search_complication" and mode == "search_ready":
			# null only the specific variants flagged by the reviewer
			for variant_name in _VARIANT_COL_TO_NAME.values():
				if r.get(variant_name):
					variant_mask = lang_mask & (out["prompt_variant"] == variant_name)
					out.loc[variant_mask, col] = None

	return out


try:
    import regex as _regex

    # Single compiled alternation — one named group per script we care about.
    # Named groups become the return value directly; lru_cache keeps each
    # codepoint from being matched more than once per process.
    # Unicode property names: Han→CJK, Hiragana/Katakana→Japanese,
    # Hangul→Korean, Oriya→Odia (Unicode uses the old name).
    _SCRIPT_RE = _regex.compile(
        r'(?P<Latin>\p{Script=Latin})'
        r'|(?P<Cyrillic>\p{Script=Cyrillic})'
        r'|(?P<Arabic>\p{Script=Arabic})'
        r'|(?P<CJK>\p{Script=Han})'
        r'|(?P<Japanese>\p{Script=Hiragana}|\p{Script=Katakana})'
        r'|(?P<Korean>\p{Script=Hangul})'
        r'|(?P<Hebrew>\p{Script=Hebrew})'
        r'|(?P<Devanagari>\p{Script=Devanagari})'
        r'|(?P<Bengali>\p{Script=Bengali})'
        r'|(?P<Tamil>\p{Script=Tamil})'
        r'|(?P<Telugu>\p{Script=Telugu})'
        r'|(?P<Kannada>\p{Script=Kannada})'
        r'|(?P<Malayalam>\p{Script=Malayalam})'
        r'|(?P<Thai>\p{Script=Thai})'
        r'|(?P<Lao>\p{Script=Lao})'
        r'|(?P<Tibetan>\p{Script=Tibetan})'
        r'|(?P<Myanmar>\p{Script=Myanmar})'
        r'|(?P<Georgian>\p{Script=Georgian})'
        r'|(?P<Ethiopic>\p{Script=Ethiopic})'
        r'|(?P<Cherokee>\p{Script=Cherokee})'
        r'|(?P<Khmer>\p{Script=Khmer})'
        r'|(?P<Mongolian>\p{Script=Mongolian})'
        r'|(?P<Armenian>\p{Script=Armenian})'
        r'|(?P<Greek>\p{Script=Greek})'
        r'|(?P<Syriac>\p{Script=Syriac})'
        r'|(?P<Thaana>\p{Script=Thaana})'
        r'|(?P<Gujarati>\p{Script=Gujarati})'
        r'|(?P<Gurmukhi>\p{Script=Gurmukhi})'
        r'|(?P<Odia>\p{Script=Oriya})'
        r'|(?P<Sinhala>\p{Script=Sinhala})'
        r'|(?P<New_Tai_Lue>\p{Script=New_Tai_Lue})'
        r'|(?P<Tai_Le>\p{Script=Tai_Le})'
        r'|(?P<Tai_Tham>\p{Script=Tai_Tham})'
        r'|(?P<Tai_Viet>\p{Script=Tai_Viet})'
        r'|(?P<Balinese>\p{Script=Balinese})'
        r'|(?P<Sundanese>\p{Script=Sundanese})'
        r'|(?P<Javanese>\p{Script=Javanese})'
        r'|(?P<Cham>\p{Script=Cham})'
        r'|(?P<Yi>\p{Script=Yi})'
        r'|(?P<Lisu>\p{Script=Lisu})'
        r'|(?P<Miao>\p{Script=Miao})'
        r'|(?P<Vai>\p{Script=Vai})'
        r'|(?P<Bamum>\p{Script=Bamum})'
        r'|(?P<Nko>\p{Script=Nko})'
        r'|(?P<Tifinagh>\p{Script=Tifinagh})'
        r'|(?P<Coptic>\p{Script=Coptic})'
        r'|(?P<Adlam>\p{Script=Adlam})'
        r'|(?P<Osage>\p{Script=Osage})'
        r'|(?P<Buginese>\p{Script=Buginese})'
        r'|(?P<Samaritan>\p{Script=Samaritan})'
        r'|(?P<Mandaic>\p{Script=Mandaic})'
    )

    @functools.lru_cache(maxsize=4096)
    def _char_script(cp: int) -> str:
        """Map a Unicode code point to a script family name."""
        m = _SCRIPT_RE.fullmatch(chr(cp))
        return m.lastgroup if m else 'Other'

except ImportError:
    def _char_script(cp: int) -> str:
        """Map a Unicode code point to a script family name (range-based fallback)."""
        if (0x0041 <= cp <= 0x007A or 0x00C0 <= cp <= 0x024F
                or 0x0250 <= cp <= 0x02AF or 0x1E00 <= cp <= 0x1EFF):
            return 'Latin'
        if 0x0400 <= cp <= 0x052F:
            return 'Cyrillic'
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
            return 'Arabic'
        if 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:
            return 'Hebrew'
        if 0x0900 <= cp <= 0x097F:
            return 'Devanagari'
        if 0x0980 <= cp <= 0x09FF:
            return 'Bengali'
        if 0x0A80 <= cp <= 0x0AFF:
            return 'Gujarati'
        if 0x0B80 <= cp <= 0x0BFF:
            return 'Tamil'
        if 0x0C00 <= cp <= 0x0C7F:
            return 'Telugu'
        if 0x0C80 <= cp <= 0x0CFF:
            return 'Kannada'
        if 0x0D00 <= cp <= 0x0D7F:
            return 'Malayalam'
        if 0x0E00 <= cp <= 0x0E7F:
            return 'Thai'
        if 0x0E80 <= cp <= 0x0EFF:
            return 'Lao'
        if 0x0F00 <= cp <= 0x0FFF:
            return 'Tibetan'
        if 0x1000 <= cp <= 0x109F:
            return 'Myanmar'
        if 0x10A0 <= cp <= 0x10FF:
            return 'Georgian'
        if (0x1100 <= cp <= 0x11FF or 0x302E <= cp <= 0x302F
                or 0xA960 <= cp <= 0xA97F or 0xAC00 <= cp <= 0xD7FF):
            return 'Korean'
        if 0x1200 <= cp <= 0x137F or 0x2D80 <= cp <= 0x2DDF:
            return 'Ethiopic'
        if 0x13A0 <= cp <= 0x13FF:
            return 'Cherokee'
        if 0x1780 <= cp <= 0x17FF:
            return 'Khmer'
        if 0x1800 <= cp <= 0x18AF:
            return 'Mongolian'
        if 0x0530 <= cp <= 0x058F:
            return 'Armenian'
        if 0x0370 <= cp <= 0x03FF:
            return 'Greek'
        if 0x0700 <= cp <= 0x074F:
            return 'Syriac'
        if 0x0780 <= cp <= 0x07BF:
            return 'Thaana'
        if 0x0A00 <= cp <= 0x0A7F:
            return 'Gurmukhi'
        if 0x0B00 <= cp <= 0x0B7F:
            return 'Odia'
        if 0x0D80 <= cp <= 0x0DFF:
            return 'Sinhala'
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF):
            return 'CJK'
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
            return 'Japanese'
        return 'Other'


def detect_dominant_script(text: str) -> str:
	"""Return the dominant writing script of a translation string.

	Ignores spaces, punctuation, and digits — only letter/mark characters
	count toward the tally. Returns 'Unknown' for empty or punctuation-only
	strings.

	Examples
	--------
	>>> detect_dominant_script('Дигитална хуманистика')
	'Cyrillic'
	>>> detect_dominant_script('Digitalna humanistika')
	'Latin'
	>>> detect_dominant_script('الإنسانيات الرقمية')
	'Arabic'
	"""
	if not text or not isinstance(text, str):
		return 'Unknown'
	counts: Counter = Counter()
	for ch in text:
		cp = ord(ch)
		cat = unicodedata.category(ch)
		# Skip spaces, punctuation, digits, control chars
		if cat[0] in ('Z', 'P', 'S', 'C') or cat == 'Nd':
			continue
		counts[_char_script(cp)] += 1
	if not counts:
		return 'Unknown'
	return counts.most_common(1)[0][0]


def detect_script_disagreement(
	translations: Dict[str, Optional[str]],
) -> Dict:
	"""Detect whether a set of translations uses more than one writing script.

	Useful for languages with digraphia (e.g. Serbian Cyrillic ↔ Latin,
	Uzbek, Azerbaijani) where the same term can be rendered in two scripts,
	inflating the apparent number of unique translations.

	Parameters
	----------
	translations : dict
		Mapping of service name → translation string (or None).

	Returns
	-------
	dict with keys:
		has_script_disagreement : bool
		scripts_found           : list of unique script names detected
		script_per_service      : list of (service, script) pairs
	"""
	valid = {s: t for s, t in translations.items()
			 if t and isinstance(t, str) and t.strip() and t not in ('nan', 'None')}
	if not valid:
		return {
			'has_script_disagreement': False,
			'scripts_found': [],
			'script_per_service': [],
		}

	per_service: List[Tuple[str, str]] = [
		(svc, detect_dominant_script(term)) for svc, term in valid.items()
	]
	unique_scripts: List[str] = sorted({s for _, s in per_service} - {'Unknown'})

	return {
		'has_script_disagreement': len(unique_scripts) > 1,
		'scripts_found': unique_scripts,
		'script_per_service': per_service,
	}


def categorize_difference(str1: str, str2: str, directionality: str = 'ltr') -> str:
	"""
	Categorize why two translation strings differ.

	Returns one of:
	  - 'identical': exact match
	  - 'capitalization': same text, different case (LTR scripts only)
	  - 'whitespace': same text ignoring whitespace
	  - 'both': differs in both capitalization and whitespace (LTR only)
	  - 'content': actual content difference

	RTL scripts (Arabic, Hebrew, etc.) skip capitalization checks since they
	typically have no case distinction.
	"""
	if str1 == str2:
		return 'identical'
	if directionality == 'ltr' and str1.lower() == str2.lower():
		return 'capitalization'
	if str1.strip().lower() == str2.strip().lower():
		if str1.strip() == str2.strip():
			return 'whitespace'
		return 'both' if directionality == 'ltr' else 'whitespace'
	return 'content'


def analyze_differences(
	service_values: Dict[str, Optional[str]],
	directionality: str = 'ltr',
) -> Dict[str, str]:
	"""
	Analyze all pairwise differences in a {key: translation} mapping.

	Returns a dict of pairwise difference types plus a 'summary' key with a
	comma-joined set of all difference types observed.
	"""
	actual = {k: v for k, v in service_values.items() if v is not None}
	if len(actual) <= 1:
		return {'summary': 'no_differences'}

	differences: Dict[str, str] = {}
	items = list(actual.items())
	for i, (k1, v1) in enumerate(items):
		for k2, v2 in items[i + 1:]:
			if v1 != v2:
				differences[f'{k1}_vs_{k2}'] = categorize_difference(v1, v2, directionality)

	if not differences:
		return {'summary': 'all_identical'}
	differences['summary'] = ','.join(sorted(set(differences.values())))
	return differences


_PLACEHOLDER_RATIONALE_STEMS = frozenset({
    'no rationale provided',
    'no rationale',
    'n/a',
    'none',
    'not applicable',
    'no explanation provided',
    'no reason provided',
})


def is_placeholder_rationale(text) -> bool:
    """Return True if *text* is a placeholder rather than a real rationale.

    LLMs sometimes return literal strings like 'No rationale provided' instead
    of an actual explanation. These should be treated the same as a missing
    rationale — they carry no keyword signal and cannot validate the translation.
    Non-string values (NaN, None, float) return False so callers don't need to
    pre-filter.
    """
    if not isinstance(text, str):
        return False
    return text.strip().lower().rstrip('.') in _PLACEHOLDER_RATIONALE_STEMS


# LLM service column pairs: (translation_col, rationale_col)
LLM_SERVICE_COL_PAIRS: List[Tuple[str, str]] = [
    ('claude_translated_term',  'claude_translation_rationale'),
    ('openai_translated_term',  'openai_translation_rationale'),
    ('gemini_translated_term',  'gemini_translation_rationale'),
    ('ollama_translated_term',  'ollama_translation_rationale'),
]


def enforce_translation_rationale_pairing(df: pd.DataFrame) -> pd.DataFrame:
    """Null out the unpaired side of any LLM translation ↔ rationale mismatch.

    For each LLM service:
    - Translation present but rationale absent/placeholder → null the translation.
    - Rationale present but translation absent → null the rationale.

    Non-LLM sources — MT baselines (GT, EasyNMT, Lingvanex) and community
    references (Wikipedia) — are skipped because they never produce rationales
    by design.

    Returns a copy with the offending cells set to NaN.
    """
    df = df.copy()
    for trans_col, rat_col in LLM_SERVICE_COL_PAIRS:
        if trans_col not in df.columns or rat_col not in df.columns:
            continue

        has_trans = df[trans_col].notna() & ~df[trans_col].astype(str).str.strip().isin(['', 'nan'])
        rat_raw   = df[rat_col].astype(str)
        has_rat   = df[rat_col].notna() & ~rat_raw.str.strip().isin(['', 'nan']) & ~rat_raw.apply(
            lambda v: is_placeholder_rationale(v)
        )

        # Translation without real rationale → drop translation
        df.loc[has_trans & ~has_rat, trans_col] = None
        # Rationale (or placeholder) without translation → drop rationale
        df.loc[~has_trans, rat_col] = None

    return df


def load_manual_exclusions(
    eval_dir: str,
) -> tuple[set, set, dict]:
    """Load manual_exclusions.csv and return three filtered structures.

    Parameters
    ----------
    eval_dir : str
        Path to the evaluation directory containing manual_exclusions.csv.

    Returns
    -------
    analysis_langs : set[str]
        Language codes excluded from all analysis (drop rows entirely).
    search_terms : set[tuple[str, str]]
        (language_code, term) pairs that must not be used as search strings.
    corrections : dict[tuple[str, str], str]
        Mapping (language_code, original_term) → corrected_term.
    """
    path = os.path.join(eval_dir, 'manual_exclusions.csv')
    if not os.path.exists(path):
        return set(), set(), {}

    df = pd.read_csv(path, dtype=str).fillna('')

    analysis_langs: set[str] = set(
        df.loc[
            (df['service'] == 'analysis_exclusion') &
            (df['language_code'].str.strip() != ''),
            'language_code',
        ]
    )
    search_terms: set[tuple[str, str]] = set(
        zip(
            df.loc[df['service'] == 'search_exclusion', 'language_code'],
            df.loc[df['service'] == 'search_exclusion', 'original_term'],
        )
    )
    corr_rows = df[
        (df['service'] == 'term_correction') &
        (df['corrected_term'].str.strip() != '')
    ]
    corrections: dict[tuple[str, str], str] = {
        (row['language_code'], row['original_term']): row['corrected_term']
        for _, row in corr_rows.iterrows()
    }

    return analysis_langs, search_terms, corrections


if __name__ == "__main__":
	 set_data_directory_path("/Users/zleblanc/CodingDH/translation_transmogrification_pipeline/datasets")
