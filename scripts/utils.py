"""
Standalone utility functions for the translation pipeline.

These are extracted from the parent CodingDH utils.py so this repo is
self-contained and can be installed independently.
"""

from __future__ import annotations

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
    try:
        return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip' if not error_bad_lines else 'error')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip' if not error_bad_lines else 'error')


def log_error_to_file(
    error_file_path: str,
    additional_data: dict,
    status_code: int,
    error_url: str,
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
    """
    error_df = pd.DataFrame([{
        "error_date": datetime.now().strftime("%Y-%m-%d"),
        "error_url": error_url,
        "status_code": status_code,
    }])
    error_df = pd.concat([error_df, pd.DataFrame([additional_data])], axis=1)

    if os.path.exists(error_file_path):
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
        error_df = read_csv_file(error_file_path)
        if 'error_date' in error_df.columns:
            error_df['error_date'] = pd.to_datetime(error_df['error_date'])
            error_df = error_df.sort_values('error_date').drop_duplicates(
                subset=drop_fields, keep='last'
            )
        elif drop_fields:
            error_df = error_df.drop_duplicates(subset=drop_fields, keep='last')
        error_df.to_csv(error_file_path, index=False)
    else:
        console.print('No error file to clean', style='bold blue')


def get_data_directory_path() -> str:
    """
    Return the data directory path from the DATA_DIR environment variable,
    falling back to a ``data/`` sibling of the repo root.

    Set ``DATA_DIR=/path/to/your/data`` in your environment or .env file.
    """
    env_path = os.environ.get('DATA_DIR')
    if env_path:
        return env_path
    # Default: data/ relative to repo root (two levels up from this file)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, 'data')


def set_data_directory_path(path: str) -> None:
	"""
	Sets data directory path.

	:param path: Path to data directory
	"""
	apikey.save("CODING_DH_DATA_DIRECTORY_PATH", path)
	console.print(f'Coding DH data directory path set to {path}', style='bold blue')

def get_data_directory_path() -> str:
	"""
	Gets data directory path.

	:return: Data directory path
	"""
	return apikey.load("CODING_DH_DATA_DIRECTORY_PATH")

_LANG_FAMILY_CACHE: Optional[Dict[str, str]] = None


def get_language_family(code: str) -> str:
    """Return the top-level family name for a language code, reading from the
    comprehensive language codes CSV. Falls back to 'Other' when unknown."""
    global _LANG_FAMILY_CACHE
    if _LANG_FAMILY_CACHE is None:
        data_dir = get_data_directory_path()
        csv_path = os.path.join(data_dir, 'metadata_files', 'language_codes_comprehensive.csv')
        if os.path.exists(csv_path):
            df = read_csv_file(csv_path)
            _LANG_FAMILY_CACHE = dict(
                zip(df['language_code'].astype(str),
                    df['family_name'].fillna('Other').astype(str))
            )
        else:
            _LANG_FAMILY_CACHE = {}
    return _LANG_FAMILY_CACHE.get(code, 'Other')


def _char_script(cp: int) -> str:
    """Map a Unicode code point to a script family name."""
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
    # CJK unified + extensions
    if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF):
        return 'CJK'
    # Japanese syllabaries (share CJK kanji, so a CJK check above catches kanji)
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


if __name__ == "__main__":
     set_data_directory_path("/Users/zleblanc/CodingDH/translation_transmogrification_pipeline/datasets")