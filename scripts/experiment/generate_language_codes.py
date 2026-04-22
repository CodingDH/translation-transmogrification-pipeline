"""
build_language_codes.py
========================
Builds a comprehensive, defensible language codes dataset by combining
four sources in a single pipeline:

1. The Unicode Common Locale Data Repository (CLDR) — Languages and Scripts ~814 language codes; authoritative for scripts, modern-use flag, official-language status, and directionality. Source: https://www.unicode.org/cldr/charts/45/supplemental/languages_and_scripts.html

2. LOC ISO 639-2 — individual language codes only (no group codes) ~190 individual language codes; Set 1 (2-letter) to Set 2 (3-letter) mapping Source: https://www.loc.gov/standards/iso639-2/php/code_list.php

3. Wikimedia language codes — languages with active Wikipedia projects ~268 codes; flags community digital presence beyond ISO standards Source: https://meta.wikimedia.org/wiki/Template:List_of_language_names_ordered_by_code

4. ISO 639-5 — language family and group codes 115 codes with full hierarchy; used to assign every language to its genetic family for downstream grouping. Source: https://en.wikipedia.org/wiki/ISO_639-5

Output files
------------
  language_codes_comprehensive.csv
    One row per language code. Columns:
      language_code       primary code (ISO 639-1 two-letter where available)
      language_name       English name
      iso639_1            two-letter code (if exists)
      iso639_2_t          three-letter terminological code (if exists)
      iso639_2_b          three-letter bibliographic code (if different)
      scripts             pipe-separated list of scripts (e.g. "Arab|Cyrl")
      primary_script      most common/modern script name
      primary_script_code ISO 15924 four-letter script code
      directionality           'rtl' or 'ltr' (authoritative; derived from CLDR primary script + FORCE_LTR overrides)
      directionality_wikimedia 'rtl', 'ltr', or '' (Wikimedia community value; useful for spotting digraphia cases like Serbian or Uzbek)
      modern_language          True/False (False = ancient, extinct, constructed)
      is_official              True if official in at least one country (CLDR)
      in_iso639_1              True if has ISO 639-1 two-letter code
      in_iso639_2              True if in ISO 639-2 individual language list
      in_wikimedia             True if has active Wikipedia project
      iso639_5_direct          direct ISO 639-5 group code (e.g. 'gem' for German)
      iso639_5_family          top-level ISO 639-5 family code (e.g. 'ine')
      family_name              English name of top-level family
      subfamily_name           English name of direct group
      sources                  pipe-separated list of contributing sources

  language_scripts_long.csv
    One row per (language x script) pair. Useful for script-switching analysis.

  iso_639_set5.csv
    115 ISO 639-5 family/group codes with hierarchy and parent info.

Usage
-----
  python build_language_codes.py
  python build_language_codes.py \\
      --cldr-file      path/to/Languages_and_Scripts.html \\
      --loc-file       path/to/loc_iso639_2.html \\
      --wikimedia-file path/to/wikimedia_codes.csv \\
      --set5-file      path/to/ISO_639-5_-_Wikipedia.html \\
      --output-dir     path/to/output/
"""

import argparse
import io
import os
import re
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup

from rich.console import Console
console = Console()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils import get_data_directory_path

# ── Shared user-agent (Wikimedia and Wikipedia require a real UA) ─────────────
def _ua():
    return {"User-Agent": "iso639-research/1.0 (contact: zleblanc@illinois.edu)"}



# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# RTL script codes (ISO 15924). Source: Unicode Bidirectional Algorithm + script property data
RTL_SCRIPT_CODES = {
    'Arab', 'Hebr', 'Thaa', 'Syrc', 'Nkoo', 'Adlm', 'Rohg', 'Tfng',
    'Mend', 'Yezi', 'Nbat', 'Palm', 'Hatr', 'Narb', 'Sarb', 'Armi',
    'Avst', 'Hung', 'Lydi', 'Ital', 'Phnx', 'Prti', 'Phli', 'Khar',
    'Cprt', 'Linb', 'Sogd',
}

# Languages whose CLDR primary script is listed as Arabic for historical reasons but whose modern standard orthography is Latin or Cyrillic. CLDR lists multiple scripts per language sorted by prevalence across all varieties; for these languages the first script reflects historical usage, not current practice. Override to ltr after script-based assignment. Sources: Ethnologue, Wikipedia orthography articles, ISO 639-5 registry.
FORCE_LTR = {
    'tr',   # Turkish     — Arabic script until 1928, Latin since
    'uz',   # Uzbek       — Arabic/Cyrillic historically, Latin in Uzbekistan since 1993
    'az',   # Azerbaijani — Arabic/Cyrillic historically, Latin in Azerbaijan since 1991
    'kk',   # Kazakh      — Arabic/Cyrillic historically, Latin transition ongoing
    'ky',   # Kyrgyz      — Cyrillic primary in Kyrgyzstan; Arabic only in diaspora
    'tk',   # Turkmen     — Latin since 1993
    'id',   # Indonesian  — Arabic script (Jawi) historical, Latin overwhelmingly primary
    'ms',   # Malay       — same as Indonesian (Jawi is secondary)
    'ha',   # Hausa       — Arabic script (Ajami) historical, Latin primary
    'so',   # Somali      — Arabic script historical, Latin official since 1972
    'ku',   # Kurdish     — Kurmanji (dominant Wikimedia variety) uses Latin
    'wo',   # Wolof       — Arabic script (Wolofal) minority, Latin primary
}

# ISO 639-2 B-code to T-code mapping (20 dual-code languages)
BCODE_TO_TCODE = {
    'alb': 'sqi', 'arm': 'hye', 'baq': 'eus', 'bur': 'mya',
    'chi': 'zho', 'cze': 'ces', 'dut': 'nld', 'fre': 'fra',
    'geo': 'kat', 'ger': 'deu', 'gre': 'ell', 'ice': 'isl',
    'mac': 'mkd', 'mao': 'mri', 'may': 'msa', 'per': 'fas',
    'rum': 'ron', 'slo': 'slk', 'tib': 'bod', 'wel': 'cym',
}

# Wikimedia section-header rows that leak into the table as data
WIKIMEDIA_JUNK = {
    'code', 'old projects', 'see also test languages',
    'closed-zh-tw', 'simple',
}

# ISO 639-2 special-purpose codes that are not real languages and should be
# excluded from the translation target set entirely.
NON_LANGUAGE_CODES = {
    'und',  # Undetermined / Unknown language
    'zxx',  # No linguistic content (e.g. music, silence)
    'mis',  # Uncoded languages (catch-all)
    'mul',  # Multiple languages
}

# Heuristic keywords for LOC group/collective codes in ISO 639-2
GROUP_KEYWORDS = [
    'languages', 'creoles', 'pidgins', 'sign languages',
    'artificial', 'miscellaneous', 'unclassified',
]

# Manual language-code to ISO 639-5 family mapping. Covers ISO 639-1 and common Wikimedia codes. Kamusella (2012) notes ISO 639-2 assumes Romanisation; this mapping uses linguistic genealogy rather than orthographic convention as the grouping principle. Where a language is a known isolate, 'isolate' is used as a sentinel so downstream users can filter appropriately.
MANUAL_LANG_TO_SET5 = {
    # Indo-European > Germanic
    'af': 'gem', 'afr': 'gem', 'da': 'gem', 'dan': 'gem',
    'de': 'gem', 'deu': 'gem', 'en': 'gem', 'eng': 'gem',
    'fy': 'gem', 'fry': 'gem', 'is': 'gem', 'isl': 'gem',
    'lb': 'gem', 'ltz': 'gem', 'nl': 'gem', 'nld': 'gem',
    'no': 'gem', 'nor': 'gem', 'nb': 'gem', 'nob': 'gem',
    'nn': 'gem', 'nno': 'gem', 'sv': 'gem', 'swe': 'gem',
    'yi': 'gem', 'yid': 'gem',
    # Wikimedia Germanic (non-ISO 639-1)
    'als': 'gem', 'ang': 'gem', 'bar': 'gem', 'got': 'gem',
    'ksh': 'gem', 'nds': 'gem', 'nds-nl': 'gem', 'pdc': 'gem',
    'sco': 'gem', 'vls': 'gem',
    # Indo-European > Romance
    'ca': 'roa', 'cat': 'roa', 'es': 'roa', 'spa': 'roa',
    'fr': 'roa', 'fra': 'roa', 'gl': 'roa', 'glg': 'roa',
    'it': 'roa', 'ita': 'roa', 'la': 'roa', 'lat': 'roa',
    'oc': 'roa', 'oci': 'roa', 'pt': 'roa', 'por': 'roa',
    'ro': 'roa', 'ron': 'roa',
    # Wikimedia Romance
    'an': 'roa', 'co': 'roa', 'ext': 'roa', 'frp': 'roa',
    'fur': 'roa', 'ht': 'roa', 'lad': 'roa', 'lij': 'roa',
    'lmo': 'roa', 'mwl': 'roa', 'nap': 'roa', 'nrm': 'roa',
    'pms': 'roa', 'roa-rup': 'roa', 'sc': 'roa', 'scn': 'roa',
    'vec': 'roa', 'wa': 'roa',
    # Indo-European > Slavic
    'be': 'sla', 'bel': 'sla', 'bg': 'sla', 'bul': 'sla',
    'bs': 'sla', 'bos': 'sla', 'cs': 'sla', 'ces': 'sla',
    'hr': 'sla', 'hrv': 'sla', 'mk': 'sla', 'mkd': 'sla',
    'pl': 'sla', 'pol': 'sla', 'ru': 'sla', 'rus': 'sla',
    'sk': 'sla', 'slk': 'sla', 'sl': 'sla', 'slv': 'sla',
    'sr': 'sla', 'srp': 'sla', 'uk': 'sla', 'ukr': 'sla',
    # Wikimedia Slavic
    'be-x-old': 'sla', 'csb': 'sla', 'cu': 'sla',
    'dsb': 'sla', 'sh': 'sla',
    # Indo-European > Baltic
    'lt': 'bat', 'lit': 'bat', 'lv': 'bat', 'lav': 'bat',
    'bat-smg': 'bat',
    # Indo-European > Celtic
    'br': 'cel', 'bre': 'cel', 'cy': 'cel', 'cym': 'cel',
    'ga': 'cel', 'gle': 'cel', 'gd': 'cel', 'gla': 'cel',
    'gv': 'cel', 'glv': 'cel', 'kw': 'cel', 'cor': 'cel',
    # Indo-European > Indic
    'as': 'inc', 'asm': 'inc', 'bn': 'inc', 'ben': 'inc',
    'gu': 'inc', 'guj': 'inc', 'hi': 'inc', 'hin': 'inc',
    'mr': 'inc', 'mar': 'inc', 'ne': 'inc', 'nep': 'inc',
    'pa': 'inc', 'pan': 'inc', 'si': 'inc', 'sin': 'inc',
    'ur': 'inc', 'urd': 'inc',
    # Wikimedia Indic
    'awa': 'inc', 'bho': 'inc', 'bpy': 'inc',
    # brx (Bodo) and dz (Dzongkha) are Tibeto-Burman, not Indo-Aryan — see sit block below
    'gbm': 'inc',
    # Indo-European > Iranian
    'fa': 'ira', 'fas': 'ira', 'ku': 'ira', 'kur': 'ira',
    'ps': 'ira', 'pus': 'ira', 'tg': 'ira', 'tgk': 'ira',
    'os': 'ira', 'oss': 'ira',
    # Wikimedia Iranian
    'ckb': 'ira', 'diq': 'ira', 'glk': 'ira',
    'khw': 'ira', 'mzn': 'ira', 'pnb': 'ira',
    # Indo-European > Greek
    'el': 'grk', 'ell': 'grk',
    # Indo-European > Armenian
    'hy': 'hyx', 'hye': 'hyx',
    # Indo-European > Albanian
    'sq': 'sqj', 'sqi': 'sqj',
    # Afro-Asiatic > Semitic
    'ar': 'sem', 'ara': 'sem', 'he': 'sem', 'heb': 'sem',
    'am': 'sem', 'amh': 'sem', 'mt': 'sem', 'mlt': 'sem',
    'ti': 'sem', 'tir': 'sem',
    'arc': 'sem', 'arz': 'sem',
    # Afro-Asiatic > Cushitic
    'so': 'cus', 'som': 'cus', 'om': 'cus', 'orm': 'cus',
    # Afro-Asiatic > Chadic
    'ha': 'cdc', 'hau': 'cdc',
    # Afro-Asiatic > Berber
    'kab': 'ber',
    # Sino-Tibetan
    'zh': 'sit', 'zho': 'sit', 'bo': 'sit', 'bod': 'sit',
    'my': 'sit', 'mya': 'sit',
    # Wikimedia Sino-Tibetan
    'brx': 'sit', 'dz': 'sit',  # Bodo and Dzongkha are Tibeto-Burman (moved from inc above)
    'cdo': 'sit', 'gan': 'sit', 'hak': 'sit', 'man': 'sit',
    'mrh': 'sit', 'wuu': 'sit', 'yue': 'sit',
    'zh-classical': 'sit', 'zh-min-nan': 'sit',
    'zh-tw': 'sit', 'zh-yue': 'sit',
    # Turkic
    'az': 'trk', 'aze': 'trk', 'ba': 'trk', 'bak': 'trk',
    'kk': 'trk', 'kaz': 'trk', 'ky': 'trk', 'kir': 'trk',
    'tk': 'trk', 'tuk': 'trk', 'tr': 'trk', 'tur': 'trk',
    'tt': 'trk', 'tat': 'trk', 'ug': 'trk', 'uig': 'trk',
    'uz': 'trk', 'uzb': 'trk',
    # Wikimedia Turkic
    'cv': 'trk', 'uz_AF': 'trk',
    # Mongolic (xal = Kalmyk is Mongolian, not Turkic — moved to xgn block below)
    # Wikimedia Mongolic
    # Uralic > Finno-Ugrian
    'et': 'fiu', 'est': 'fiu', 'fi': 'fiu', 'fin': 'fiu',
    'hu': 'fiu', 'hun': 'fiu',
    # Uralic > Sami
    'se': 'smi', 'sme': 'smi',
    # Wikimedia Uralic
    'fiu-vro': 'fiu', 'kv': 'fiu', 'udm': 'fiu',
    # Dravidian
    'kn': 'dra', 'kan': 'dra', 'ml': 'dra', 'mal': 'dra',
    'ta': 'dra', 'tam': 'dra', 'te': 'dra', 'tel': 'dra',
    # Niger-Kordofanian > Bantu
    'sw': 'bnt', 'swa': 'bnt', 'zu': 'bnt', 'zul': 'bnt',
    'xh': 'bnt', 'xho': 'bnt', 'rw': 'bnt', 'kin': 'bnt',
    'ny': 'bnt', 'nya': 'bnt', 'sn': 'bnt', 'sna': 'bnt',
    'st': 'bnt', 'sot': 'bnt', 'tn': 'bnt', 'tsn': 'bnt',
    'ln': 'bnt', 'lin': 'bnt', 'lg': 'bnt', 'lug': 'bnt',
    'kg': 'bnt', 'kon': 'bnt', 'rn': 'bnt', 'run': 'bnt',
    'ss': 'bnt', 'ssw': 'bnt', 'ts': 'bnt', 've': 'bnt',
    'nso': 'bnt', 'tum': 'bnt',
    # Niger-Kordofanian > broader Niger-Congo
    'yo': 'nic', 'yor': 'nic', 'ig': 'nic', 'ibo': 'nic',
    'ak': 'nic', 'aka': 'nic', 'ff': 'nic', 'ful': 'nic',
    'ee': 'nic', 'ewe': 'nic', 'tw': 'nic', 'bm': 'nic',
    # Nilo-Saharan
    'kr': 'ssa', 'kau': 'ssa', 'sg': 'ssa', 'sag': 'ssa',
    # Austronesian
    'id': 'map', 'ind': 'map', 'ms': 'map', 'msa': 'map',
    'mg': 'map', 'mlg': 'map', 'mi': 'map', 'mri': 'map',
    'tl': 'map', 'tgl': 'map', 'fj': 'map', 'fij': 'map',
    'sm': 'map', 'smo': 'map', 'to': 'map', 'ton': 'map',
    'bi': 'map', 'bis': 'map', 'ch': 'map', 'cha': 'map',
    'mh': 'map', 'mah': 'map', 'na': 'map', 'nau': 'map',
    # Wikimedia Austronesian
    'bcl': 'map', 'bug': 'map', 'ceb': 'map', 'gil': 'map',
    'haw': 'map', 'ilo': 'map', 'jv': 'map', 'map-bms': 'map',
    'min': 'map', 'pag': 'map', 'pam': 'map', 'su': 'map',
    'tet': 'map', 'ty': 'map', 'war': 'map',
    # Japonic
    'ja': 'jpx', 'jpn': 'jpx',
    # Tai-Kadai
    'th': 'tai', 'tha': 'tai', 'lo': 'tai', 'lao': 'tai',
    # Austro-Asiatic
    'km': 'mkh', 'khm': 'mkh', 'vi': 'mkh', 'vie': 'mkh',
    # Mongolic
    'mn': 'xgn', 'mon': 'xgn', 'bxr': 'xgn', 'xal': 'xgn',
    # Eskimo-Aleut
    'iu': 'esx', 'iku': 'esx',
    # Indigenous Americas
    'ay': 'sai', 'qu': 'sai', 'gn': 'sai',
    'cr': 'nai', 'ik': 'nai', 'kl': 'nai',
    'nv': 'nai', 'oj': 'nai',
    # Wikimedia Indigenous Americas
    'cho': 'nai', 'chr': 'nai', 'chy': 'nai',
    'mus': 'nai', 'nah': 'nai',
    # Constructed / Artificial
    'eo': 'art', 'epo': 'art', 'ia': 'art', 'ina': 'art',
    'io': 'art', 'ido': 'art', 'vo': 'art', 'vol': 'art',
    'jbo': 'art', 'ie': 'art',
    'tlh': 'art', 'tokipona': 'art',
    # Caucasian
    'ka': 'ccs', 'kat': 'ccs',        # Georgian > South Caucasian
    'ab': 'ccn', 'abk': 'ccn',        # Abkhaz > North Caucasian
    'av': 'ccn', 'ava': 'ccn',        # Avaric
    'ce': 'ccn', 'che': 'ccn',        # Chechen
    'inh': 'ccn', 'lzz': 'ccn',       # Wikimedia Caucasian
    'xmf': 'ccs',                      # Megrelian > South Caucasian
    # Basque — has its own ISO 639-5 code (euq) as a family of one
    'eu': 'euq', 'eus': 'euq',
    # Korean — language isolate (no genetic relatives)
    'ko': 'isolate', 'kor': 'isolate',
    # Creole / Mixed
    'pap': 'crp', 'pih': 'crp', 'tpi': 'crp',
    # Sign languages
    'sgn': 'sgn',
}


# ══════════════════════════════════════════════════════════════════════════════
# PARSER 1 — Unicode CLDR: Languages and Scripts
# ══════════════════════════════════════════════════════════════════════════════

def parse_cldr_html(source: str) -> tuple:
    """
    Parses the CLDR Languages and Scripts HTML page. Returns (lang_df, long_df) where lang_df is one row per language and long_df is one row per (language x script) pair.
    
    CLDR is the primary source for language codes, and the other sources are merged in as supplements. CLDR is authoritative for script assignments, modern-use flag, official-language status, and directionality. The parser is designed to be robust to changes in the HTML structure; it identifies the relevant table by looking for the one with the most rows, and it handles both 7-column rows (new language) and 2- or 3-column rows (additional scripts for the same language).
    
    Parameters:
    -----------
    source: str
		Local file path or live URL of the CLDR Languages and Scripts HTML page.
    
    Returns:
    --------
    Tuple of (lang_df, long_df):
    - lang_df: DataFrame with one row per language code, containing aggregated information about scripts, modern-use status, official status, and directionality.
    - long_df: DataFrame with one row per (language x script) pair, containing detailed information about each script used for each language.
    """
    is_file = os.path.exists(source)
    if is_file:
        with open(source, 'r', encoding='utf-8') as f:
            content = f.read()
        console.print(f"  Loaded CLDR file: {source}")
    else:
        resp = requests.get(source, headers=_ua(), timeout=20)
        resp.raise_for_status()
        content = resp.text
        console.print(f"  Fetched CLDR URL: {source}")

    soup = BeautifulSoup(content, 'html.parser')
    table = max(soup.find_all('table'), key=lambda t: len(t.find_all('tr')))
    rows  = table.find_all('tr')

    long_records = []
    current_lang = current_code = current_ml = current_p = None

    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        n = len(cells)
        if n == 7:
            current_lang, current_code = cells[0], cells[1]
            current_ml,   current_p   = cells[2], cells[3]
            script_name, script_code, script_modern = cells[4], cells[5], cells[6]
        elif n == 3 and current_lang:
            script_name, script_code, script_modern = cells[0], cells[1], cells[2]
        elif n == 2 and current_lang:
            script_name, script_code, script_modern = cells[0], cells[1], ''
        else:
            continue
        if not current_code:
            continue
        long_records.append({
            'language_name':  current_lang,
            'language_code':  current_code,
            'modern_language': current_ml != 'O',
            'is_official':    current_p != 'N',
            'script_name':    script_name,
            'script_code':    script_code,
            'script_modern':  script_modern != 'N',
            'is_rtl':         script_code in RTL_SCRIPT_CODES,
        })

    long_df = pd.DataFrame(long_records)

    def agg_language(grp):
        modern_mask = grp['script_modern']
        pidx = modern_mask.idxmax() if modern_mask.any() else grp.index[0]
        return pd.Series({
            'language_name':        grp['language_name'].iloc[0],
            'modern_language':      grp['modern_language'].iloc[0],
            'is_official':          grp['is_official'].iloc[0],
            'scripts':              '|'.join(grp['script_name'].tolist()),
            'script_codes':         '|'.join(grp['script_code'].tolist()),
            'primary_script':       grp.loc[pidx, 'script_name'],
            'primary_script_code':  grp.loc[pidx, 'script_code'],
            'directionality':       'rtl' if grp['is_rtl'].any() else 'ltr',
            'n_scripts':            len(grp),
        })

    lang_df = (long_df.groupby('language_code')
               .apply(agg_language, include_groups=False)
               .reset_index())

    console.print(f"  CLDR: {len(lang_df)} codes | "
        f"{lang_df['modern_language'].sum()} modern | "
        f"{(lang_df['directionality']=='rtl').sum()} RTL")
    return lang_df, long_df


# ══════════════════════════════════════════════════════════════════════════════
# PARSER 2 — LOC ISO 639-2
# ══════════════════════════════════════════════════════════════════════════════

def parse_loc_html(source: str) -> pd.DataFrame:
    """
    Parse LOC ISO 639-2 HTML. Returns DataFrame with iso639_1 / iso639_2_t / iso639_2_b mappings.
    
    LOC ISO 639-2 is the primary source for ISO 639-2 codes and the mapping from ISO 639-1 two-letter codes to ISO 639-2 three-letter codes. The parser is designed to be robust to changes in the HTML structure; it identifies the relevant table by looking for the first one with at least 3 columns, and it handles both single-code rows (one code for both bibliographic and terminological) and dual-code rows (separate B and T codes). It also filters out group codes based on heuristic keywords in the language name, since ISO 639-2 includes both individual languages and groups/collectives.
    
    Parameters:
    -----------
    source: str
		Local file path or live URL of the LOC ISO 639-2 HTML page.
        
    Returns:
    --------
    DataFrame with columns:
		- iso639_1: two-letter code (if exists)	
		- iso639_2_t: three-letter terminological code (if exists)
		- iso639_2_b: three-letter bibliographic code (if different from T code)
		- english_name: English name of the language
		- english_name_primary: primary English name (first before any semicolon)
		- is_group: True if the code represents a group/collective rather than an individual language (heuristic based on keywords in the name)

    """
    is_file = os.path.exists(source)
    if is_file:
        with open(source, 'r', encoding='latin-1') as f:
            content = f.read()
        console.print(f"  Loaded LOC file: {source}")
    else:
        resp = requests.get(source, headers=_ua(), timeout=20)
        resp.raise_for_status()
        content = resp.text
        console.print(f"  Fetched LOC URL: {source}")

    soup  = BeautifulSoup(content, 'html.parser')
    table = soup.find_all('table')[0]
    rows  = table.find_all('tr')

    records = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cells) < 3:
            continue
        raw = cells[0].strip()
        if not raw or raw.lower().startswith('library'):
            continue
        b_match = re.match(r'^([a-z]{3})\s*\(B\)([a-z]{3})\s*\(T\)$', raw)
        if b_match:
            iso2_b, iso2_t = b_match.group(1), b_match.group(2)
        elif re.match(r'^[a-z]{3}$', raw):
            iso2_t = raw
            iso2_b = {v: k for k, v in BCODE_TO_TCODE.items()}.get(raw, '')
        elif raw == 'qaa-qtz':
            continue
        else:
            continue
        name = cells[2].strip()
        records.append({
            'iso639_2_t':           iso2_t,
            'iso639_2_b':           iso2_b,
            'iso639_1':             cells[1].strip(),
            'english_name':         name,
            'english_name_primary': name.split(';')[0].strip(),
            'is_group':             any(kw in name.lower() for kw in GROUP_KEYWORDS),
        })

    df = (pd.DataFrame(records)
          .drop_duplicates(subset='iso639_2_t')
          .reset_index(drop=True))
    ind = (~df['is_group']).sum()
    console.print(f"  LOC: {ind} individual languages | "
        f"{df['is_group'].sum()} groups | "
        f"{(df['iso639_1'].str.len()==2).sum()} with ISO 639-1 codes")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PARSER 3 — Wikimedia language codes
# ══════════════════════════════════════════════════════════════════════════════

def parse_wikimedia(source: str) -> pd.DataFrame:
    """
    Load Wikimedia language codes from local CSV or live URL.
    
    Wikimedia is the primary source for identifying which languages have active Wikipedia projects, which is a strong signal of modern usage and online presence. The parser can handle both a local CSV file (e.g. downloaded from Wikimedia) or a live URL pointing to the Wikimedia language code list. It filters out non-language codes based on known junk entries and code length, and it standardizes the column names for merging with the other sources.
    
    Parameters:
    -----------
    source: str
		Local file path or live URL of the Wikimedia language codes CSV or HTML page.
        
    Returns:
    --------
    DataFrame with columns:
        - language_code:            Wikimedia language code (usually ISO 639-1 or ISO 639-3)
        - language_name:            English name of the language as listed by Wikimedia
        - directionality_wikimedia: 'rtl' or 'ltr' normalised from Wikimedia's own value, or '' if unavailable
        - local_name:               Local name of the language as listed by Wikimedia (if available)
    """
    is_file = os.path.exists(source) and source.endswith('.csv')
    if is_file:
        df = pd.read_csv(source).rename(columns={
            'code': 'language_code',
            'English language name': 'language_name',
            'local language name': 'local_name',
        })
        console.print(f"  Loaded Wikimedia file: {source}")
    else:
        resp = requests.get(source, headers=_ua(), timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        df = pd.read_html(io.StringIO(str(soup.find_all('table')[0])))[0]
        df.columns = ['language_code', 'language_name', 'directionality','local_name', 'wikipedia_article', 'comment']
        console.print(f"  Fetched Wikimedia URL: {source}")

    df['language_code'] = df['language_code'].astype(str).str.strip()
    df = df[~df['language_code'].str.lower().isin(WIKIMEDIA_JUNK)]
    df = df[df['language_code'].str.len() >= 2]
    df = df.drop_duplicates(subset='language_code').reset_index(drop=True)
    # Normalise directionality to ltr/rtl lowercase if present
    if 'directionality' in df.columns:
        df['directionality_wikimedia'] = (
            df['directionality'].astype(str).str.strip().str.lower()
            .map(lambda x: 'rtl' if 'rtl' in x else ('ltr' if 'ltr' in x else ''))
        )
    else:
        df['directionality_wikimedia'] = ''
    console.print(f"  Wikimedia: {len(df)} codes with active Wikipedia projects | "
                  f"{(df['directionality_wikimedia']=='rtl').sum()} RTL")
    cols = ['language_code', 'language_name', 'directionality_wikimedia']
    if 'local_name' in df.columns:
        cols.append('local_name')
    return df[cols]


# ══════════════════════════════════════════════════════════════════════════════
# PARSER 4 — ISO 639-5 language families (Wikipedia)
# ══════════════════════════════════════════════════════════════════════════════

def parse_iso639_5(source: str) -> pd.DataFrame:
    """
    Parse ISO 639-5 Wikipedia table.
    
    ISO 639-5 is the primary source for language family/group codes, which are important for understanding the genealogical relationships between languages and for grouping languages that may not have individual codes. The parser is designed to be robust to changes in the HTML structure; it identifies the relevant table by looking for the one with the most rows, and it handles the hierarchical structure of the families based on the formatting of the first column. It also cleans up the codes and names by removing any footnote markers or extraneous whitespace.
    
    Parameters:
	-----------
	source: str
		Local file path or live URL of the Wikipedia ISO 639-5 page.
    
    Returns:
    --------
    DataFrame with columns:
	- iso639_5: ISO 639-5 three-letter code for the family/group
	- iso639_2: ISO 639-2 code if the family has one (some families have a corresponding ISO 639-2 code, but many do not)
	- family_name: English name of the family/group
	- hierarchy: Original text from the first column showing the hierarchical structure (e.g. "Indo-European: Germanic: West Germanic")
	- parent_code: ISO 639-5 code of the immediate parent family/group (empty for top-level families)
	- depth: Integer representing the depth in the hierarchy (0 for top- level families, 1 for their immediate subgroups, etc.)
	- notes: Any additional notes from the table (e.g. "language isolate" for families of one, or "contains only extinct languages" for defunct families)
    """
    is_file = os.path.exists(source)
    if is_file:
        with open(source, 'r', encoding='utf-8') as f:
            content = f.read()
        console.print(f"  Loaded ISO 639-5 file: {source}")
    else:
        resp = requests.get(source, headers=_ua(), timeout=20)
        resp.raise_for_status()
        content = resp.text
        console.print(f"  Fetched ISO 639-5 URL: {source}")

    soup  = BeautifulSoup(content, 'html.parser')
    table = max(soup.find_all('table'), key=lambda t: len(t.find_all('tr')))
    rows  = table.find_all('tr')

    records = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        if len(cells) < 4:
            continue
        code  = re.sub(r'\[.*?\]', '', cells[1].strip()).strip()
        iso2  = re.sub(r'\[.*?\]', '', cells[2].strip()).strip()
        if not re.match(r'^[a-z]{3}$', code):
            continue
        hier  = cells[0].strip()
        parts = [p.strip() for p in hier.split(':') if p.strip()]
        ancs  = [p for p in parts if p != code]
        records.append({
            'iso639_5':    code,
            'iso639_2':    iso2 if re.match(r'^[a-z]{3}$', iso2) else '',
            'family_name': cells[3].strip(),
            'hierarchy':   hier,
            'parent_code': ancs[-1] if ancs else '',
            'depth':       len(ancs),
            'notes':       cells[4].strip() if len(cells) > 4 else '',
        })

    df = (pd.DataFrame(records)
          .drop_duplicates(subset='iso639_5')
          .sort_values(['depth', 'iso639_5'])
          .reset_index(drop=True))
    console.print(f"  ISO 639-5: {len(df)} codes "
        f"({(df['depth']==0).sum()} top-level, "
        f"{(df['depth']>0).sum()} subgroups)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MERGE STEP 1 — Combine CLDR + LOC + Wikimedia into comprehensive language code DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def build_comprehensive(
    cldr_df: pd.DataFrame,
    loc_df:  pd.DataFrame,
    wiki_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the three primary sources into one authoritative DataFrame.
    
    Parameters:
    -----------
    cldr_df: DataFrame from parse_cldr_html, with one row per language code and aggregated information about scripts, modern-use status, official status, and directionality.
	loc_df: DataFrame from parse_loc_html, with ISO 639-1 and ISO 639-2 code mappings and language names.
    wiki_df: DataFrame from parse_wikimedia, with Wikimedia language codes, names, and directionality_wikimedia.

    Returns:
    --------
    Merged DataFrame with one row per language code, containing:
    - language_code:            primary code (CLDR, supplemented by LOC and Wikimedia)
    - language_name:            English name (CLDR preferred, then LOC, then Wikimedia)
    - modern_language:          True if marked modern in CLDR; True for LOC/Wikimedia-only codes
    - is_official:              True if official in at least one country (CLDR); False for LOC/Wikimedia-only
    - directionality:           authoritative 'rtl'/'ltr' derived from CLDR primary script + FORCE_LTR overrides;
                                for wiki-only codes with no CLDR data, falls back to directionality_wikimedia
    - directionality_wikimedia: Wikimedia community value ('rtl', 'ltr', or ''); preserved for all codes
                                that appear in Wikimedia, enabling comparison with the CLDR-derived value
    - n_scripts:                number of scripts from CLDR (0 for LOC/Wikimedia-only codes)
    - iso639_2_t:               ISO 639-2 terminological code (from LOC)
    - iso639_2_b:               ISO 639-2 bibliographic code (from LOC)
    - iso639_1:                 ISO 639-1 two-letter code (from LOC)
    - is_group_iso639_2:        True if the ISO 639-2 code is a group/collective rather than individual language
    - in_wikimedia:             True if the code has an active Wikipedia project
    - in_iso639_1:              True if the code has an ISO 639-1 two-letter form
    - in_iso639_2:              True if the code appears in the LOC ISO 639-2 individual language list
    - sources:                  pipe-separated contributing sources (e.g. 'cldr|wikimedia', 'wikimedia_only')
    """

    merged = cldr_df.copy()
    merged['sources'] = 'cldr'

    # Add Wikimedia-only codes (not in CLDR)
    wiki_only = wiki_df[~wiki_df['language_code'].isin(merged['language_code'])].copy()
    if len(wiki_only):
        for col in merged.columns:
            if col not in wiki_only.columns:
                wiki_only[col] = (True if col == 'modern_language'
                                  else False if col == 'is_official'
                                  else 0 if col == 'n_scripts'
                                  else '')
        # Use Wikimedia's own directionality for wiki-only codes; fall back to ltr
        wiki_only['directionality'] = wiki_only.get('directionality_wikimedia', '').replace('', 'ltr').fillna('ltr')
        wiki_only['sources'] = 'wikimedia_only'
        merged = pd.concat([merged, wiki_only[merged.columns]], ignore_index=True)
        console.print(f"  Added {len(wiki_only)} Wikimedia-only codes")

    # Add LOC individual language codes not in CLDR or Wikimedia
    loc_ind  = loc_df[~loc_df['is_group']].copy()
    loc_only = loc_ind[
        ~loc_ind['iso639_2_t'].isin(merged['language_code']) &
        ~loc_ind['iso639_1'].isin(merged['language_code'])
    ].copy()
    if len(loc_only):
        loc_rows = []
        for _, r in loc_only.iterrows():
            code = r['iso639_1'] if r['iso639_1'] else r['iso639_2_t']
            loc_rows.append({c: (True if c == 'modern_language'
                                 else False if c == 'is_official'
                                 else 0 if c == 'n_scripts'
                                 else ('loc_only' if c == 'sources'
                                       else (r['english_name_primary']
                                             if c == 'language_name'
                                             else (code if c == 'language_code'
                                                   else ''))))
                              for c in merged.columns})
            for _, r in loc_only.iterrows():
                code = r['iso639_1'] if r['iso639_1'] else r['iso639_2_t']
                row_dict = {c: '' for c in merged.columns}
                row_dict.update({
                    'language_code': code,
                    'language_name': r['english_name_primary'],
                    'modern_language': True, 'is_official': False,
                    'directionality': 'ltr', 'n_scripts': 0,
                    'sources': 'loc_only',
                })
                loc_rows.append(row_dict)
            break  # inner loop handles all rows
        merged = pd.concat([merged, pd.DataFrame(loc_rows[1:])], ignore_index=True)
        console.print(f"  Added {len(loc_only)} LOC-only codes")

    # Build ISO 639 code lookups
    loc_by_1 = (loc_ind[loc_ind['iso639_1'].str.len() == 2]
                .set_index('iso639_1')[['iso639_2_t', 'iso639_2_b', 'is_group']])
    loc_by_2 = loc_ind.set_index('iso639_2_t')[['iso639_1', 'iso639_2_b', 'is_group']]

    def get_loc(code):
        code = str(code)
        if len(code) == 2 and code in loc_by_1.index:
            r = loc_by_1.loc[code]
            return r['iso639_2_t'], r['iso639_2_b'], '', r['is_group']
        if len(code) == 3 and code in loc_by_2.index:
            r = loc_by_2.loc[code]
            return code, r['iso639_2_b'], r['iso639_1'], r['is_group']
        if len(code) == 2:
            m = loc_ind[loc_ind['iso639_1'] == code]
            if not m.empty:
                r = m.iloc[0]
                return r['iso639_2_t'], r['iso639_2_b'], code, r['is_group']
        return '', '', '', False

    results = [get_loc(c) for c in merged['language_code']]
    merged['iso639_2_t']        = [r[0] for r in results]
    merged['iso639_2_b']        = [r[1] for r in results]
    merged['iso639_1']          = [r[2] for r in results]
    merged['is_group_iso639_2'] = [r[3] for r in results]

    # 2-letter codes ARE iso639_1 codes — fill in where blank
    mask2 = merged['language_code'].str.len() == 2
    merged.loc[mask2 & (merged['iso639_1'] == ''), 'iso639_1'] = \
        merged.loc[mask2 & (merged['iso639_1'] == ''), 'language_code']

    # Boolean membership flags
    wiki_codes = set(wiki_df['language_code'].astype(str))
    merged['in_wikimedia'] = merged['language_code'].isin(wiki_codes)
    merged['in_iso639_1']  = merged['iso639_1'].str.len() == 2
    merged['in_iso639_2']  = merged['iso639_2_t'].str.len() == 3

    # Update sources column
    def build_src(row):
        srcs = [row['sources']]
        if row['in_iso639_1'] and 'loc' not in srcs[0]:
            srcs.append('iso639_1')
        if row['in_iso639_2'] and 'loc' not in srcs[0]:
            srcs.append('iso639_2')
        if row['in_wikimedia'] and 'wikimedia' not in srcs[0]:
            srcs.append('wikimedia')
        return '|'.join(srcs)
    merged['sources'] = merged.apply(build_src, axis=1)

    # Merge Wikimedia directionality and local_name onto all rows (including CLDR codes)
    wiki_cols = ['language_code', 'directionality_wikimedia']
    if 'local_name' in wiki_df.columns:
        wiki_cols.append('local_name')
    merged = merged.merge(wiki_df[wiki_cols], on='language_code', how='left')
    merged['directionality_wikimedia'] = merged['directionality_wikimedia'].fillna('')
    if 'local_name' in merged.columns:
        merged['local_name'] = merged['local_name'].fillna('')

    # Authoritative directionality from CLDR primary script code
    merged['directionality'] = merged['primary_script_code'].apply(
        lambda c: 'rtl' if str(c) in RTL_SCRIPT_CODES else 'ltr'
    )
    # Override languages whose CLDR primary script is historical, not current
    merged.loc[merged['language_code'].isin(FORCE_LTR), 'directionality'] = 'ltr'

    final_cols = [
        'language_code', 'language_name', 'local_name',
        'iso639_1', 'iso639_2_t', 'iso639_2_b',
        'scripts', 'script_codes',
        'primary_script', 'primary_script_code',
        'directionality', 'directionality_wikimedia',
        'modern_language', 'is_official', 'n_scripts',
        'in_iso639_1', 'in_iso639_2', 'in_wikimedia',
        'is_group_iso639_2', 'sources',
    ]
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = ''
    return (merged[final_cols]
            .drop_duplicates(subset='language_code')
            .loc[lambda d: ~d['language_code'].isin(NON_LANGUAGE_CODES)]
            .sort_values('language_code')
            .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# MERGE STEP 2 — Add ISO 639-5 family hierarchy
# ══════════════════════════════════════════════════════════════════════════════

def add_family_info(comp_df: pd.DataFrame, set5_df: pd.DataFrame,
                    fallback_json: str = None) -> pd.DataFrame:
    """
    Join ISO 639-5 family information onto the comprehensive DataFrame.

    Uses MANUAL_LANG_TO_SET5 as the primary lookup, then falls back to
    language_family_assignments.json for codes that MANUAL does not cover.
    The JSON file also corrects known errors in legacy reference data (e.g.
    Limburgish misclassified as Romance, Zhuang as Sino-Tibetan).

    Parameters
    ----------
    comp_df : DataFrame from build_comprehensive.
    set5_df : DataFrame containing ISO 639-5 family/group codes and hierarchy.
    fallback_json : path to language_family_assignments.json; defaults to the
        file sitting alongside this script.

    Returns
    -------
    comp_df enriched with iso639_5_direct, iso639_5_family, family_name,
    and subfamily_name columns.
    """
    import json as _json

    code_to_parent = dict(zip(set5_df['iso639_5'], set5_df['parent_code']))
    code_to_name   = dict(zip(set5_df['iso639_5'], set5_df['family_name']))
    code_to_depth  = dict(zip(set5_df['iso639_5'], set5_df['depth']))

    # Correct known scraped-name quirks in the ISO 639-5 Wikipedia table
    code_to_name.update({
        'crp': 'Creoles and pidgins',   # scraped as "Creolesandpidgins" (missing spaces)
        'sgn': 'Sign languages',         # scraped as lowercase "sign languages"
        'jpx': 'Japanese languages',     # scraped as "Japanese (family)"
        'euq': 'Basque languages',       # scraped as "Basque (family)"
    })

    # Load JSON fallback (codes not covered by MANUAL_LANG_TO_SET5)
    if fallback_json is None:
        fallback_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..', 'datasets', 'metadata_files',
                                     'language_family_assignments.json')
    json_fallback: dict = {}
    if os.path.exists(fallback_json):
        with open(fallback_json, encoding='utf-8') as f:
            json_fallback = {e['language_code']: e for e in _json.load(f)}

    def top_family(code):
        if not code or code == 'isolate':
            return code
        visited, current = set(), code
        while current and current not in visited:
            if code_to_depth.get(current, 99) == 0:
                return current
            visited.add(current)
            current = code_to_parent.get(current, '')
        return code

    rows_out = []
    for _, row in comp_df.iterrows():
        lang = str(row['language_code']).strip()
        iso2 = str(row.get('iso639_2_t', '')).strip()
        direct = (MANUAL_LANG_TO_SET5.get(lang) or
                  MANUAL_LANG_TO_SET5.get(iso2) or '')

        # JSON fallback for codes not in MANUAL
        fb = json_fallback.get(lang, {})
        if not direct and fb.get('iso639_5'):
            direct = fb['iso639_5']

        if not direct and lang in code_to_name:
            direct = lang

        top    = top_family(direct) if direct else ''
        fname  = ('Language isolate' if top == 'isolate'
                  else code_to_name.get(top, ''))
        sfname = code_to_name.get(direct, fname)

        # Final fallback: use family_name string from JSON when ISO 639-5 gives nothing
        if not fname and fb.get('family_name'):
            fname  = fb['family_name']
            sfname = fb.get('family_name', fname)

        rows_out.append((direct, top, fname, sfname))

    comp_df = comp_df.copy()
    comp_df['iso639_5_direct'] = [r[0] for r in rows_out]
    comp_df['iso639_5_family'] = [r[1] for r in rows_out]
    comp_df['family_name']     = [r[2] for r in rows_out]
    comp_df['subfamily_name']  = [r[3] for r in rows_out]

    covered = (comp_df['iso639_5_family'] != '').sum()
    console.print(f"  Family assigned: {covered}/{len(comp_df)} "
        f"({covered/len(comp_df)*100:.0f}%)")
    json_covered = sum(1 for e in rows_out if e[2])
    console.print(f"  JSON fallback used: {sum(1 for lang in comp_df['language_code'] if lang in json_fallback and not MANUAL_LANG_TO_SET5.get(lang))} codes")
    return comp_df


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC LOADER — used by the translation pipeline
# ══════════════════════════════════════════════════════════════════════════════

def load_language_codes(output_dir: str = None) -> pd.DataFrame:
    """
    Load the comprehensive language codes CSV, generating it first if absent.

    Returns all ~858 language codes from ``language_codes_comprehensive.csv``
    (special-purpose non-language codes like ``und`` and ``zxx`` are excluded
    during CSV generation). Column names are normalised for the translation
    pipeline:

      language_code, directionality, English language name,
      local language name, comment, local or English Wikipedia article

    plus all other columns from the CSV. Use the ``in_wikimedia``,
    ``in_iso639_2``, and ``modern_language`` columns to apply your own
    downstream filters.
    """
    if output_dir is None:
        output_dir = os.path.join(get_data_directory_path(), 'metadata_files')
    csv_path = os.path.join(output_dir, 'language_codes_comprehensive.csv')

    if not os.path.exists(csv_path):
        main(
            cldr_source='https://www.unicode.org/cldr/charts/45/supplemental/languages_and_scripts.html',
            loc_source='https://www.loc.gov/standards/iso639-2/php/code_list.php',
            wikimedia_source='https://meta.wikimedia.org/wiki/Template:List_of_language_names_ordered_by_code',
            set5_source='https://en.wikipedia.org/wiki/ISO_639-5',
            output_dir=output_dir,
        )

    df = pd.read_csv(csv_path)

    # Patch family_name gaps in existing CSVs without requiring a full regeneration.
    # add_family_info() handles this during generation; this covers pre-existing files.
    # Names are derived from the ISO 639-5 hierarchy where possible so they match the
    # authoritative source; the JSON family_name string is only used for codes that have
    # no ISO 639-5 entry (isolates, undeciphered scripts, etc.).
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'datasets', 'metadata_files',
                             'language_family_assignments.json')
    if os.path.exists(json_path):
        import json as _json
        with open(json_path, encoding='utf-8') as f:
            json_fallback = {e['language_code']: e for e in _json.load(f)}

        # Build ISO 639-5 hierarchy lookup (same corrections as add_family_info)
        set5_path = os.path.join(output_dir, 'iso_639_set5.csv')
        if os.path.exists(set5_path):
            set5_df = pd.read_csv(set5_path)
            _c2parent = dict(zip(set5_df['iso639_5'], set5_df['parent_code']))
            _c2name   = dict(zip(set5_df['iso639_5'], set5_df['family_name']))
            _c2depth  = dict(zip(set5_df['iso639_5'], set5_df['depth']))
            _c2name.update({
                'crp': 'Creoles and pidgins',
                'sgn': 'Sign languages',
                'jpx': 'Japanese languages',
                'euq': 'Basque languages',
            })
        else:
            _c2parent = _c2name = _c2depth = {}

        def _top_name(iso5):
            visited, current = set(), iso5
            while current and current not in visited:
                if _c2depth.get(current, 99) == 0:
                    return _c2name.get(current, '')
                visited.add(current)
                current = _c2parent.get(str(current), '')
            return _c2name.get(iso5, '')

        missing_mask = df['family_name'].isna() | (df['family_name'] == '')
        for idx, row in df[missing_mask].iterrows():
            fb = json_fallback.get(str(row['language_code']).strip(), {})
            if fb.get('iso639_5'):
                fname = _top_name(fb['iso639_5'])
                if fname:
                    df.at[idx, 'family_name'] = fname
                    continue
            if fb.get('family_name'):
                df.at[idx, 'family_name'] = fb['family_name']

    df = df.rename(columns={
        'language_name': 'English language name',
        'local_name': 'local language name',
    })
    for col in ('local language name', 'comment', 'local or English Wikipedia article'):
        if col not in df.columns:
            df[col] = ''
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(cldr_source, loc_source, wikimedia_source, set5_source, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(get_data_directory_path(), 'metadata_files')
    os.makedirs(output_dir, exist_ok=True)

    console.print("\n[bold cyan]Step 1/5 — Unicode CLDR: Languages and Scripts[/bold cyan]")
    cldr_lang_df, cldr_long_df = parse_cldr_html(cldr_source)

    console.print("\n[bold cyan]Step 2/5 — LOC ISO 639-2[/bold cyan]")
    loc_df = parse_loc_html(loc_source)

    console.print("\n[bold cyan]Step 3/5 — Wikimedia language codes[/bold cyan]")
    wiki_df = parse_wikimedia(wikimedia_source)

    console.print("\n[bold cyan]Step 4/5 — ISO 639-5: Language families[/bold cyan]")
    set5_df = parse_iso639_5(set5_source)

    console.print("\n[bold cyan]Step 5/5 — Merging and writing outputs[/bold cyan]")
    comprehensive = build_comprehensive(cldr_lang_df, loc_df, wiki_df)
    comprehensive = add_family_info(comprehensive, set5_df)

    p_main = os.path.join(output_dir, 'language_codes_comprehensive.csv')
    comprehensive.to_csv(p_main, index=False)
    console.print(f"  ✓ {p_main} ({len(comprehensive)} rows)")

    p_long = os.path.join(output_dir, 'language_scripts_long.csv')
    cldr_long_df.to_csv(p_long, index=False)
    console.print(f"  ✓ {p_long} ({len(cldr_long_df)} script pairs)")

    p_set5 = os.path.join(output_dir, 'iso_639_set5.csv')
    set5_df.to_csv(p_set5, index=False)
    console.print(f"  ✓ {p_set5} ({len(set5_df)} family codes)")

    console.print("\n[bold]Summary[/bold]")
    c = comprehensive
    console.print(f"  Total language codes:         {len(c)}")
    console.print(f"  ISO 639-1 (2-letter):         {c['in_iso639_1'].sum()}")
    console.print(f"  ISO 639-2 (individual):       {c['in_iso639_2'].sum()}")
    console.print(f"  Wikimedia (active Wikipedia): {c['in_wikimedia'].sum()}")
    console.print(f"  Modern languages:             {c['modern_language'].sum()}")
    console.print(f"  Ancient/extinct/constructed:  {(~c['modern_language']).sum()}")
    console.print(f"  RTL languages:                {(c['directionality']=='rtl').sum()}")
    console.print(f"  With family assigned:         {(c['iso639_5_family']!='').sum()}")
    console.print(f"\n  Top 10 families by code count:")
    top = (c[c['family_name'] != ''].groupby('family_name')
           .size().sort_values(ascending=False).head(10))
    for name, n in top.items():
        console.print(f"    {name:<44} {n}")

    console.print("\n[bold green]Sources and rationale:[/bold green]")
    console.print("  CLDR:      script data, modern-use flag, official status (widest coverage)")
    console.print("  LOC:       ISO 639-1/2 code standardisation (Kamusella 2012 critique noted)")
    console.print("  Wikimedia: proxy for digital community presence beyond ISO")
    console.print("  ISO 639-5: linguistic family grouping for downstream analysis")

    return comprehensive, cldr_long_df, set5_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build comprehensive language codes: CLDR + LOC + Wikimedia + ISO 639-5"
    )
    parser.add_argument('--cldr-file',
        default='https://www.unicode.org/cldr/charts/45/supplemental/languages_and_scripts.html')
    parser.add_argument('--loc-file',
        default='https://www.loc.gov/standards/iso639-2/php/code_list.php')
    parser.add_argument('--wikimedia-file',
        default='https://meta.wikimedia.org/wiki/Template:List_of_language_names_ordered_by_code')
    parser.add_argument('--set5-file',
        default='https://en.wikipedia.org/wiki/ISO_639-5')
    parser.add_argument('--output-dir',
        default=None,
        help='Output directory (default: metadata_files/ under DATA_DIR)')
    args = parser.parse_args()
    main(args.cldr_file, args.loc_file, args.wikimedia_file,
         args.set5_file, args.output_dir)