"""
generate_language_codes.py
===========================
Builds a comprehensive, defensible language codes dataset by combining
five authoritative sources in a single pipeline:

1. Unicode CLDR (version 48.2, via npm cldr-core + cldr-localenames-full)
   ~802 language codes. Authoritative for scripts, directionality,
   modern-use status, and official-language status. Downloaded from the
   npm registry and cached locally — no HTML scraping.
   Cite: Unicode CLDR 48.2.0, https://github.com/unicode-org/cldr-json

2. LOC ISO 639-2 — individual language codes only (no group codes)
   ~190 codes; provides the 2-letter ↔ 3-letter mapping.
   Source: https://www.loc.gov/standards/iso639-2/php/code_list.php

3. Wikimedia language codes — languages with active Wikipedia projects
   ~268 codes; proxy for community digital presence beyond ISO standards.
   Source: https://meta.wikimedia.org/wiki/Template:List_of_language_names_ordered_by_code

4. CLDR v45 supplement — 23 ISO 639-3 languages present in CLDR 45 but
   dropped in CLDR 48.2 because their communities stopped submitting locale
   data (fonts, date formats, etc.) required for software internationalisation.
   That criterion does not apply to this pipeline (scholarly translation).
   All 23 are valid, living ISO 639-3 languages retained here to preserve
   coverage. Marked sources='cldr_v45'. See CLDR_V45_SUPPLEMENT constant.

5. SIL ISO 639-3 name table — English reference names for the ~190 CLDR
   codes absent from CLDR's en/languages.json. CLDR's English locale-names
   file covers ~693 of its ~802 codes; the rest fell through to code-as-name
   via en_names.get(code, code). This step patches those rows using SIL's
   ~7,900-code reference table. Two codes not in SIL (kro, tokipona) are
   filled from the _SIL_NAME_OVERRIDES constant. The table is downloaded
   once and cached alongside the CLDR npm packages.

Family hierarchy (Step 4) also comes from CLDR's languageGroups.json,
which lives in the same cldr-core npm package fetched in Step 1.

Output files
------------
  language_codes_comprehensive.csv
    One row per language code. Key columns:
      language_code            primary code (ISO 639-1 two-letter where available)
      language_name            English name (from CLDR)
      iso639_1                 two-letter code (if exists)
      iso639_2_t               three-letter terminological code (if exists)
      iso639_2_b               three-letter bibliographic code (if different)
      scripts                  pipe-separated script list (e.g. "Arab|Cyrl")
      primary_script           most-used modern script name
      primary_script_code      ISO 15924 four-letter script code
      directionality           'rtl' or 'ltr' (derived from CLDR primary script)
      directionality_wikimedia 'rtl', 'ltr', or '' (Wikimedia value; useful for digraphia cases)
      modern_language          True/False (False = ancient, extinct, constructed)
      is_official              True if official in at least one country (CLDR territoryInfo)
      in_iso639_1              True if has an ISO 639-1 two-letter code
      in_iso639_2              True if in LOC ISO 639-2 individual language list
      in_wikimedia             True if has an active Wikipedia project
      iso639_5_direct          direct CLDR/ISO 639-5 group code (e.g. 'gem' for German)
      iso639_5_family          top-level family code (e.g. 'ine')
      family_name              English name of top-level family
      subfamily_name           English name of direct group
      cldr_version             provenance tag for the CLDR source:
                               '48.2.0' (or whichever --cldr-version was passed) for rows
                               from the live CLDR JSON fetch; '45' for the 23 CLDR v45
                               supplement codes; '' for Wikimedia-only and LOC-only rows
                               that carry no CLDR data at all.
      sources                  pipe-separated contributing sources

  language_scripts_long.csv
    One row per (language × script) pair. Useful for script-switching analysis.

  iso_639_set5.csv
    CLDR language group codes with parent and depth info.

Usage
-----
  python generate_language_codes.py
  python generate_language_codes.py \\
      --cldr-version   48.2.0 \\
      --cldr-cache-dir /path/to/cache \\
      --loc-file       https://www.loc.gov/standards/iso639-2/php/code_list.php \\
      --wikimedia-file path/to/wikimedia_codes.csv \\
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
from scripts.experiment.parse_cldr_json import (
    parse_cldr_json as _parse_cldr_json,
    parse_cldr_family_groups as _parse_cldr_family_groups,
)

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

# A real language code starts with a letter and contains only letters, digits,
# hyphens, and underscores (max 15 chars). This rejects Wikimedia section-header
# strings like "see also: test languages at the Wikimedia Incubator" that slip
# through the WIKIMEDIA_JUNK set-membership filter, and also rejects float NaN.
_VALID_CODE_RE = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{1,14}$')


def _is_valid_language_code(code) -> bool:
    if not isinstance(code, str):
        return False
    return bool(_VALID_CODE_RE.match(code))


# ── BCP 47 identifier context ─────────────────────────────────────────────────
# `language_code` remains this project's stable row identifier. The BCP 47 layer
# below is a parallel interoperability hint for web/API contexts; it never
# changes row identity and intentionally does NOT expand the dataset into
# locale variants (en-US, en-GB, es-MX, etc.). The unit of analysis for Coding
# DH is language/community discovery, not locale-specific localization.

def _canonicalize_bcp47_like_tag(tag: str) -> str:
    """Return a conventionally cased BCP 47-style tag, or ''.

    This is intentionally lightweight and dependency-free. It does not claim to
    validate against the full IANA Language Subtag Registry; it applies common
    BCP 47 casing (language lower, script title, region upper) and a few
    deprecated-subtag preferences. For full validation, downstream users may
    install the optional ``langcodes`` package.
    """
    if tag is None:
        return ''
    tag = str(tag).strip().replace('_', '-')
    if not tag or tag.lower() in {'none', 'null'}:
        return ''
    parts = [p for p in tag.split('-') if p]
    if not parts:
        return ''
    # Preserve the real language code 'nan' (Min Nan Chinese); never treat as missing.
    primary = parts[0].lower()
    primary = DEPRECATED_LANGUAGE_SUBTAG_PREFERRED.get(primary, primary)
    canon = [primary]
    for i, subtag in enumerate(parts[1:], start=1):
        low = subtag.lower()
        if len(low) == 1:
            # Private use / extension singletons: lower-case from here on
            canon.append(low)
            canon.extend(p.lower() for p in parts[i+1:])
            break
        if len(subtag) == 4 and subtag.isalpha():
            # Script subtag (Latn, Cyrl, Arab, Hant): title case
            canon.append(subtag.title())
        elif (len(subtag) == 2 and subtag.isalpha()) or (len(subtag) == 3 and subtag.isdigit()):
            # Region subtag: upper case (US, GB) or UN M.49 numeric (419)
            canon.append(subtag.upper())
        else:
            canon.append(low)
    return '-'.join(canon)


def _preferred_bcp47_for_row(row):
    """Choose a preferred BCP 47-style tag for a language row.

    Returns (tag, source, note). language_code remains canonical; this tag is
    an interoperability hint for web/HTML/XML/API contexts.
    """
    code = str(row.get('language_code', '')).strip()
    if code in WIKIMEDIA_TO_BCP47_OVERRIDES:
        return (_canonicalize_bcp47_like_tag(WIKIMEDIA_TO_BCP47_OVERRIDES[code]),
                'manual_override',
                f'project/Wikimedia code {code} mapped to preferred BCP 47-style tag')
    # Prefer ISO 639-1 — IANA / BCP 47 uses the shortest primary subtag available
    iso1 = str(row.get('iso639_1', '')).strip()
    if iso1 and iso1.lower() not in {'nan', 'none'}:
        return (_canonicalize_bcp47_like_tag(iso1), 'iso639_1', '')
    # Then ISO 639-2/T (terminological) — closer to modern ISO 639-3 practice
    iso2t = str(row.get('iso639_2_t', '')).strip()
    if iso2t and iso2t.lower() not in {'nan', 'none'}:
        return (_canonicalize_bcp47_like_tag(iso2t), 'iso639_2_t', '')
    # Finally, normalize the project code itself
    return (_canonicalize_bcp47_like_tag(code), 'language_code', '')


def _candidate_codes_for_services(row) -> list:
    """Return likely codes/tags to test against service-specific support lists."""
    values = [
        row.get('bcp47_tag', ''),
        row.get('language_code', ''),
        row.get('wikimedia_code', ''),
        row.get('iso639_1', ''),
        row.get('iso639_2_t', ''),
        row.get('iso639_2_b', ''),
    ]
    out = []
    for v in values:
        if v is None: continue
        v = str(v).strip()
        if not v or v.lower() in {'none', 'null'}: continue
        variants = {v, v.replace('_', '-'), _canonicalize_bcp47_like_tag(v)}
        # Add legacy aliases for API compatibility (iw for he, jw for jv, etc.)
        for legacy, preferred in DEPRECATED_LANGUAGE_SUBTAG_PREFERRED.items():
            if _canonicalize_bcp47_like_tag(v) == preferred:
                variants.add(legacy)
        for cand in variants:
            if cand and cand not in out:
                out.append(cand)
    return out


# Wikimedia legacy / community codes mapped to preferred IANA-registry BCP 47 tags.
# These are pure interoperability hints — language_code itself is unchanged.
WIKIMEDIA_TO_BCP47_OVERRIDES = {
    # Chinese Wikimedia legacy/community codes with preferred modern subtags
    'zh-min-nan':   'nan',      # Min Nan Chinese (also protects the literal code nan)
    'zh-yue':       'yue',      # Cantonese
    'zh-classical': 'lzh',      # Literary/Classical Chinese
    'zh-tw':        'zh-TW',    # Taiwan/Traditional Chinese project code
    # Wikimedia / CLDR locale-style codes
    'uz_AF':        'uz-AF',    # Uzbek in Afghanistan (Perso-Arabic script)
    'nds-nl':       'nds-NL',   # Low Saxon as used in the Netherlands
    # Orthography / community variants
    'be-x-old':     'be-tarask',  # Belarusian Taraškievica orthography
    # Wikimedia historical / composite project codes with clearer IANA primary subtags
    'bat-smg':      'sgs',      # Samogitian
    'fiu-vro':      'vro',      # Võro
    'map-bms':      'jv',       # Banyumasan (no stable ISO 639-3 code in common use)
    'roa-rup':      'rup',      # Aromanian
    # Constructed / community code used by some platforms
    'tokipona':     'tok',      # Toki Pona (IANA / ISO 639-3 subtag)
}

# Deprecated language subtags still seen in some APIs, mapped to current preferred.
DEPRECATED_LANGUAGE_SUBTAG_PREFERRED = {
    'iw': 'he',  # Hebrew (Google still accepts iw)
    'ji': 'yi',  # Yiddish
    'in': 'id',  # Indonesian
    'jw': 'jv',  # Javanese
    'mo': 'ro',  # Moldovan → Romanian
}

# Default filename for the optional service-support CSV consumed by
# add_service_language_codes(). Lives in datasets/metadata_files/.
SERVICE_LANGUAGE_CODE_SUPPORT_FILENAME = 'service_language_code_support.csv'


# ISO 639-2 special-purpose codes that are not real languages and should be
# excluded from the translation target set entirely.
NON_LANGUAGE_CODES = {
    'und',  # Undetermined / Unknown language
    'zxx',  # No linguistic content (e.g. music, silence)
    'mis',  # Uncoded languages (catch-all)
    'mul',  # Multiple languages
}

# SIL ISO 639-3 reference name table — used by enrich_language_names_from_sil()
# to fill in the ~190 CLDR codes that have no entry in CLDR's en/languages.json.
_SIL_URL = 'https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab'

# Hardcoded names for codes not covered by SIL's ISO 639-3 registry:
#   kro      — CLDR group code for Kru languages (ISO 639-5); no SIL ISO 639-3 entry
#   tokipona — constructed language (Toki Pona); never received an ISO 639-3 code
_SIL_NAME_OVERRIDES = {
    'kro':      'Kru languages',
    'tokipona': 'Toki Pona',
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
    'be-x-old': 'sla', 'crn': 'sla', 'csb': 'sla', 'cu': 'sla',
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
    # CLDR-only codes not covered by ISO 639-1/2/Wikimedia lookups above
    # Semitic (Afro-Asiatic)
    'apd': 'sem',   # Sudanese Spoken Arabic
    'mey': 'sem',   # Hassaniyya (Saharan Arabic variety)
    # Berber (Afro-Asiatic)
    'grr': 'ber',   # Taznatit (Algerian Berber)
    'mzb': 'ber',   # Tumzabt (Mzab Berber, Algeria)
    # Niger-Kordofanian
    'bjt': 'nic',   # Balanta-Ganja (Atlantic-Congo, Senegal/Guinea-Bissau)
    'bsc': 'nic',   # Bassari (Atlantic-Congo, Senegal/Guinea-Bissau)
    'bsq': 'nic',   # Basa (Cameroon, Niger-Congo)
    'knf': 'nic',   # Mankanya (Atlantic-Congo, Guinea-Bissau)
    'mfv': 'nic',   # Mandjak (Atlantic-Congo, Guinea-Bissau)
    'snf': 'nic',   # Noon (Senegambian, Senegal)
    'tnr': 'bnt',   # Ménik/Tonga (Bantu, Zambia)
    # Indo-European — Romance
    'lld': 'roa',   # Ladin (Romance, spoken in South Tyrol)
    # Indo-European — Germanic
    'mhn': 'gem',   # Mòcheno (Germanic, spoken in Trentino)
    # Indo-European — Indic
    'knn': 'inc',   # Konkani (Indo-Aryan, India)
    # Indo-European — Greek
    'ecy': 'grk',   # Cypriot Greek (extinct variety)
    'gmy': 'grk',   # Mycenaean Greek (ancient)
    # Central American Indian
    'ccr': 'cai',   # Cacaopera (Misumalpan, El Salvador)
    'len': 'cai',   # Lenca (Honduras/El Salvador; sometimes treated as isolate)
    # Uto-Aztecan (North American)
    'ppl': 'azc',   # Pipil / Nawat (Uto-Aztecan, El Salvador)
    # Hmong-Mien
    'mww': 'hmx',   # Hmong Daw / White Hmong
    # Sino-Tibetan
    'nan': 'zhx',   # Min Nan Chinese / Taiwanese
    'stu': 'tbq',   # Samtao (Tibeto-Burman, Myanmar)
    'suz': 'tbq',   # Sunwar (Tibeto-Burman, Nepal)
    # CLDR v45 supplement — dropped from CLDR 48.2 for administrative reasons only
    'ase': 'sgn',   # American Sign Language
    'cwd': 'nai',   # Woods Cree
    'dzg': 'ssa',   # Dazaga (Nilo-Saharan)
    'gom': 'inc',   # Goan Konkani (Indo-Aryan)
    'hax': 'isolate',  # Southern Haida (language isolate)
    'hdn': 'isolate',  # Northern Haida (language isolate)
    'ike': 'esx',   # Eastern Canadian Inuktitut
    'kbl': 'ssa',   # Kanembu (Nilo-Saharan)
    'lou': 'crp',   # Louisiana Creole
    'lsm': 'bnt',   # Saamia (Bantu, Uganda)
    'mde': 'ssa',   # Maba (Nilo-Saharan)
    'mye': 'bnt',   # Myene (Bantu, Gabon)
    'ojb': 'nai',   # Northwestern Ojibwa
    'ojc': 'nai',   # Central Ojibwa
    'ojg': 'nai',   # Eastern Ojibwa
    'sba': 'ssa',   # Ngambay (Nilo-Saharan)
    'shu': 'sem',   # Chadian Arabic (Semitic, Arabic script)
    'slh': 'nai',   # Southern Lushootseed
    'str': 'nai',   # Straits Salish
    'tce': 'nai',   # Southern Tutchone
    'tgx': 'nai',   # Tagish
    'tht': 'nai',   # Tahltan
    'ttm': 'nai',   # Northern Tutchone
}

# 23 ISO 639-3 languages that were in CLDR 45 but dropped from CLDR 48.2 because their
# communities stopped submitting Core locale data (font samples, date formats, etc.).
# CLDR's removal criterion applies to software internationalisation, not scholarly
# translation. All codes are valid living languages retained here for pipeline coverage.
# Source: CLDR 45 language_codes_comprehensive.csv (git 7c9ad3c).
# directionality is corrected where CLDR 45 had 'Zzzz' (Unknown Script): shu uses Arabic.
CLDR_V45_SUPPLEMENT = [
    # (code,  English name,                   directionality, modern, is_official)
    ('ase', 'American Sign Language',           'ltr', True, True),
    ('cwd', 'Woods Cree',                       'ltr', True, True),
    ('dzg', 'Dazaga',                           'ltr', True, True),
    ('gom', 'Goan Konkani',                     'ltr', True, False),
    ('hax', 'Southern Haida',                   'ltr', True, True),
    ('hdn', 'Northern Haida',                   'ltr', True, True),
    ('ike', 'Eastern Canadian Inuktitut',       'ltr', True, True),
    ('kbl', 'Kanembu',                          'ltr', True, True),
    ('lou', 'Louisiana Creole',                 'ltr', True, True),
    ('lsm', 'Saamia',                           'ltr', True, True),
    ('mde', 'Maba',                             'ltr', True, True),
    ('mye', 'Myene',                            'ltr', True, True),
    ('ojb', 'Northwestern Ojibwa',              'ltr', True, True),
    ('ojc', 'Central Ojibwa',                   'ltr', True, True),
    ('ojg', 'Eastern Ojibwa',                   'ltr', True, True),
    ('sba', 'Ngambay',                          'ltr', True, True),
    ('shu', 'Chadian Arabic',                   'rtl', True, True),
    ('slh', 'Southern Lushootseed',             'ltr', True, True),
    ('str', 'Straits Salish',                   'ltr', True, True),
    ('tce', 'Southern Tutchone',                'ltr', True, True),
    ('tgx', 'Tagish',                           'ltr', True, True),
    ('tht', 'Tahltan',                          'ltr', True, True),
    ('ttm', 'Northern Tutchone',                'ltr', True, True),
]


# ══════════════════════════════════════════════════════════════════════════════
# PARSER 1 — LOC ISO 639-2
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
        - language_code: Wikimedia language code (usually ISO 639-1 or ISO 639-3)
        - language_name: English name of the language as listed by Wikimedia
        - directionality_wikimedia: 'rtl' or 'ltr' normalised from Wikimedia's own value, or '' if unavailable
        - local_name: Local name of the language as listed by Wikimedia (if available)
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
    df = df[df['language_code'].apply(_is_valid_language_code)]
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
# MERGE STEP 1 — Combine CLDR + LOC + Wikimedia into comprehensive language code DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def build_comprehensive(
    cldr_df: pd.DataFrame,
    loc_df:  pd.DataFrame,
    wiki_df: pd.DataFrame,
    cldr_version: str = '',
) -> pd.DataFrame:
    """
    Merge CLDR, LOC, and Wikimedia into one authoritative DataFrame.

    Parameters
    ----------
    cldr_df : DataFrame from _parse_cldr_json(), one row per language code with aggregated script, modern-use, official-status, and directionality data.
    loc_df : DataFrame from parse_loc_html(), with ISO 639-1/2 code mappings.
    wiki_df : DataFrame from parse_wikimedia(), with Wikimedia codes and directionality.
    cldr_version : version string passed through to the output column (e.g. '48.2.0'). Used for provenance tracking; see cldr_version column notes below.

    Returns
    -------
    Merged DataFrame with one row per language code. Key columns:

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
    - cldr_version:             provenance tag — the cldr_version argument for CLDR rows; '' for
                                Wikimedia-only and LOC-only rows (they carry no CLDR data).
                                The CLDR v45 supplement added by add_cldr_v45_supplement() sets this to '45'.
    """

    merged = cldr_df.copy()
    merged['sources'] = 'cldr'
    merged['cldr_version'] = cldr_version

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
    # Wikimedia-only codes with no CLDR script data: trust Wikimedia's directionality
    # (e.g. uz_AF — Afghan Uzbek uses Perso-Arabic script, correctly marked rtl by Wikimedia)
    _no_script = merged['primary_script_code'].astype(str).isin({'', 'nan', 'None'})
    merged.loc[_no_script & (merged['directionality_wikimedia'] == 'rtl'), 'directionality'] = 'rtl'

    final_cols = [
        'language_code', 'language_name', 'local_name',
        'iso639_1', 'iso639_2_t', 'iso639_2_b',
        'scripts', 'script_codes',
        'primary_script', 'primary_script_code',
        'directionality', 'directionality_wikimedia',
        'modern_language', 'is_official', 'n_scripts',
        'in_iso639_1', 'in_iso639_2', 'in_wikimedia',
        'is_group_iso639_2', 'sources', 'cldr_version',
    ]
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = ''
    return (merged[final_cols]
            .drop_duplicates(subset='language_code')
            .loc[lambda d: ~d['language_code'].isin(NON_LANGUAGE_CODES)]
            .loc[lambda d: d['language_code'].apply(_is_valid_language_code)]
            .sort_values('language_code')
            .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# MERGE STEP 1b — Re-add CLDR v45 supplement codes
# ══════════════════════════════════════════════════════════════════════════════

def add_cldr_v45_supplement(comp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the 23 ISO 639-3 languages from CLDR_V45_SUPPLEMENT that are not
    already present in comp_df.

    These codes were in CLDR 45 but removed from CLDR 48.2 when their locale
    communities stopped submitting Core data. The removal reason is
    administrative (no software localisation contributors), not linguistic.
    They are retained here for scholarly translation coverage and marked
    sources='cldr_v45' so downstream analysis can filter or annotate them.

    If a future CLDR version re-adds a code, the CLDR 48.2 row takes
    precedence — the supplement row is skipped for any code already present.
    """
    existing = set(comp_df['language_code'].astype(str))
    new_rows = []
    for code, name, direction, modern, official in CLDR_V45_SUPPLEMENT:
        if code in existing:
            continue
        row = {c: '' for c in comp_df.columns}
        row.update({
            'language_code':   code,
            'language_name':   name,
            'directionality':  direction,
            'modern_language': modern,
            'is_official':     official,
            'n_scripts':       0,
            'in_iso639_1':     False,
            'in_iso639_2':     False,
            'in_wikimedia':    False,
            'is_group_iso639_2': False,
            'sources':         'cldr_v45',
            'cldr_version':    '45',
        })
        new_rows.append(row)

    if not new_rows:
        return comp_df

    result = (pd.concat([comp_df, pd.DataFrame(new_rows)], ignore_index=True)
              .sort_values('language_code')
              .reset_index(drop=True))
    console.print(f"  Added {len(new_rows)} CLDR v45 supplement codes (sources='cldr_v45')")
    return result


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
    from scripts.experiment.parse_cldr_json import _ISO639_5_NAMES

    code_to_parent = dict(zip(set5_df['iso639_5'], set5_df['parent_code']))
    code_to_name   = dict(zip(set5_df['iso639_5'], set5_df['family_name']))
    code_to_depth  = dict(zip(set5_df['iso639_5'], set5_df['depth']))

    # Supplement with names for ISO 639-5 codes used in MANUAL_LANG_TO_SET5 that
    # are not in CLDR's languageGroups (e.g. hyx=Armenian, sqj=Albanian, grk=Greek).
    # These appear as iso639_5_direct values but have no entry in set5_df, so
    # top_family() returns the code itself and we need a name for it.
    code_to_name.update({k: v for k, v in _ISO639_5_NAMES.items()
                         if k not in code_to_name or not code_to_name[k]})

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
# MERGE STEP 3 — Fill in missing English names from SIL ISO 639-3
# ══════════════════════════════════════════════════════════════════════════════

def _load_sil_names(cache_dir: str) -> dict:
    """Download and cache the SIL ISO 639-3 name table. Returns {code: Ref_Name}."""
    import io as _io
    cache_path = os.path.join(cache_dir, 'iso-639-3.tab')
    if not os.path.exists(cache_path):
        os.makedirs(cache_dir, exist_ok=True)
        console.print('  Fetching SIL ISO 639-3 name table...')
        resp = requests.get(_SIL_URL, headers=_ua(), timeout=30)
        resp.raise_for_status()
        with open(cache_path, 'wb') as f:
            f.write(resp.content)
        console.print(f'  Cached to {cache_path}')
    sil = pd.read_csv(cache_path, sep='\t', dtype=str, na_filter=False)
    return dict(zip(sil['Id'].str.strip(), sil['Ref_Name'].str.strip()))


def enrich_language_names_from_sil(comp_df: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
    """
    Fill in English names for language codes that fell through CLDR's en/languages.json.

    CLDR ships English names for ~693 of its ~802 language codes; the remaining
    ~190 CLDR codes (plus a handful from other sources) fall through to the
    ``en_names.get(code, code)`` default in parse_cldr_json.py, leaving the code
    itself as the language_name. This function patches those rows using SIL's ISO
    639-3 reference name table (~7,900 individual language codes), then applies
    _SIL_NAME_OVERRIDES for the two codes not covered by SIL (kro, tokipona).
    """
    missing_mask = comp_df['language_name'] == comp_df['language_code']
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        return comp_df

    sil_names = _load_sil_names(cache_dir)
    sil_names.update(_SIL_NAME_OVERRIDES)

    comp_df = comp_df.copy()
    filled = 0
    for idx in comp_df[missing_mask].index:
        code = comp_df.at[idx, 'language_code']
        name = sil_names.get(code)
        if name:
            comp_df.at[idx, 'language_name'] = name
            filled += 1

    console.print(f'  SIL name enrichment: {filled}/{n_missing} missing names filled')
    still_missing = comp_df.loc[
        comp_df['language_name'] == comp_df['language_code'], 'language_code'
    ].tolist()
    if still_missing:
        console.print(f'  Still unnamed: {still_missing}')
    return comp_df


# ══════════════════════════════════════════════════════════════════════════════
# MERGE STEP — Identifier context (BCP 47 + service support)
# ══════════════════════════════════════════════════════════════════════════════

def add_identifier_context(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Add BCP 47 and platform-code context columns without changing row identity.

    `language_code` remains the project-level stable identifier. This function
    adds an interoperability layer alongside it:

    - `project_language_code`: explicit duplicate of the project identifier
    - `wikimedia_code`: exact Wikimedia code where applicable
    - `bcp47_tag`: preferred BCP 47-style web / interchange tag
    - `bcp47_source`: provenance ('iso639_1' / 'iso639_2_t' / 'manual_override' / 'language_code')
    - `bcp47_note`: free-text note for overrides
    - `bcp47_is_project_code`: bool — true when bcp47_tag matches the normalized project code
    - `service_code_candidates`: pipe-separated candidate codes for service matching

    Does NOT create new rows for regional/script variants. That expansion would
    change the analytical unit from language/community to locale.
    """
    df = comp_df.copy()
    df['project_language_code'] = df['language_code'].astype(str)

    # Preserve exact Wikimedia project codes separately.
    df['wikimedia_code'] = ''
    if 'in_wikimedia' in df.columns:
        mask = df['in_wikimedia'].astype(bool)
        df.loc[mask, 'wikimedia_code'] = df.loc[mask, 'language_code'].astype(str)

    chosen = df.apply(_preferred_bcp47_for_row, axis=1)
    df['bcp47_tag']    = [x[0] for x in chosen]
    df['bcp47_source'] = [x[1] for x in chosen]
    df['bcp47_note']   = [x[2] for x in chosen]
    df['bcp47_is_project_code'] = (
        df['bcp47_tag'] ==
        df['language_code'].astype(str)
            .str.replace('_', '-', regex=False)
            .map(_canonicalize_bcp47_like_tag)
    )
    df['service_code_candidates'] = df.apply(
        lambda row: '|'.join(_candidate_codes_for_services(row)), axis=1
    )

    n_diff = int((df['bcp47_tag'] != df['language_code'].astype(str)).sum())
    n_over = int((df['bcp47_source'] == 'manual_override').sum())
    console.print(f'  Identifier context: {n_diff} rows have BCP47 tag different from project code; '
                  f'{n_over} manual overrides')
    return df


def add_service_language_codes(comp_df: pd.DataFrame, support_csv: str = None) -> pd.DataFrame:
    """Map rows to exact codes required by services/APIs when a support CSV exists.

    The optional CSV must contain at least `service` and `service_code` columns.
    Other columns (language_name, support_level, model, notes) are ignored by
    the matcher but preserved in the source CSV for documentation.

    For each distinct service, adds:
        <service>_code       — exact supported code selected for this row (or '')
        <service>_supported  — boolean

    Matching uses bcp47_tag, language_code, wikimedia_code, ISO fields, and
    legacy aliases. Never modifies language_code itself.
    """
    df = comp_df.copy()
    if support_csv is None:
        # default lookup: metadata_files/service_language_code_support.csv
        from pathlib import Path
        here = Path(__file__).resolve()
        # generate_language_codes.py lives at scripts/experiment/; metadata is sibling-of-parent
        candidate = here.parent.parent.parent / 'datasets' / 'metadata_files' / SERVICE_LANGUAGE_CODE_SUPPORT_FILENAME
        support_csv = str(candidate) if candidate.exists() else None
    if not support_csv or not os.path.exists(support_csv):
        console.print(f'  Service support CSV not found; skipping service-code columns')
        return df

    support = pd.read_csv(support_csv, dtype=str, na_filter=False)
    required = {'service', 'service_code'}
    if not required.issubset(support.columns):
        console.print(f'  Service support CSV missing required columns {required}: {support_csv}')
        return df

    support['service'] = support['service'].str.strip()
    support['service_code'] = support['service_code'].str.strip()

    for service, sdf in support.groupby('service'):
        if not service: continue
        # Lookup: exact code → exact code, plus underscore→hyphen and canonical variants
        code_lookup = {c: c for c in sdf['service_code'] if c}
        for c in list(code_lookup):
            code_lookup.setdefault(c.replace('_', '-'), c)
            code_lookup.setdefault(_canonicalize_bcp47_like_tag(c), c)
        # Per-row match via candidate codes
        selected = []
        for _, row in df.iterrows():
            match = ''
            for cand in _candidate_codes_for_services(row):
                if cand in code_lookup:
                    match = code_lookup[cand]
                    break
            selected.append(match)

        safe = re.sub(r'[^0-9A-Za-z_]+', '_', service).strip('_').lower()
        df[f'{safe}_code']      = selected
        df[f'{safe}_supported'] = [bool(x) for x in selected]
        console.print(f'  Service codes: {service} matched {sum(bool(x) for x in selected)}/{len(df)} rows')

    return df


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
        main(output_dir=output_dir)

    # converters={'language_code': str} bypasses pandas' default NA detection for
    # this column only, preserving 'nan' (Min Nan Chinese, ISO 639-3) as a string
    # rather than converting it to float NaN.
    df = pd.read_csv(csv_path, converters={'language_code': str})

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

    # Patch language_name gaps in existing CSVs without requiring a full regeneration.
    # enrich_language_names_from_sil() handles this during generation; this covers
    # pre-existing files where CLDR's en/languages.json fallback left codes as names.
    _name_missing = df['language_name'] == df['language_code']
    if _name_missing.any():
        try:
            _sil_cache = os.path.join(output_dir, 'cldr-cache')
            _sil = _load_sil_names(_sil_cache)
            _sil.update(_SIL_NAME_OVERRIDES)
            for _idx in df[_name_missing].index:
                _code = df.at[_idx, 'language_code']
                _name = _sil.get(_code)
                if _name:
                    df.at[_idx, 'language_name'] = _name
        except Exception:
            pass

    df = df.rename(columns={
        'language_name': 'English language name',
        'local_name': 'local language name',
    })
    for col in ('local language name', 'comment', 'local or English Wikipedia article'):
        if col not in df.columns:
            df[col] = ''
    # Drop null codes and junk strings that may have slipped into the CSV before
    # this validation was added (e.g. "see also: test languages at the Wikimedia
    # Incubator", or the string 'nan' read back as float NaN from Min Nan Chinese).
    df = df[df['language_code'].apply(_is_valid_language_code)].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(cldr_version='48.2.0', cldr_cache_dir=None,
         loc_source=None, wikimedia_source=None,
         output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(get_data_directory_path(), 'metadata_files')
    os.makedirs(output_dir, exist_ok=True)

    if cldr_cache_dir is None:
        cldr_cache_dir = os.path.join(output_dir, 'cldr-cache')
    if loc_source is None:
        loc_source = 'https://www.loc.gov/standards/iso639-2/php/code_list.php'
    if wikimedia_source is None:
        wikimedia_source = 'https://meta.wikimedia.org/wiki/Template:List_of_language_names_ordered_by_code'

    console.print(f"\n[bold cyan]Step 1/5 — Unicode CLDR {cldr_version} (JSON)[/bold cyan]")
    cldr_lang_df, cldr_long_df = _parse_cldr_json(cldr_version, cldr_cache_dir)

    console.print("\n[bold cyan]Step 2/5 — LOC ISO 639-2[/bold cyan]")
    loc_df = parse_loc_html(loc_source)

    console.print("\n[bold cyan]Step 3/5 — Wikimedia language codes[/bold cyan]")
    wiki_df = parse_wikimedia(wikimedia_source)

    console.print(f"\n[bold cyan]Step 4/5 — CLDR language family groups {cldr_version}[/bold cyan]")
    set5_df = _parse_cldr_family_groups(cldr_version, cldr_cache_dir)

    console.print("\n[bold cyan]Step 5/5 — SIL ISO 639-3 English name enrichment[/bold cyan]")

    console.print("\n[bold cyan]Merging and writing outputs[/bold cyan]")
    comprehensive = build_comprehensive(cldr_lang_df, loc_df, wiki_df, cldr_version=cldr_version)
    comprehensive = add_cldr_v45_supplement(comprehensive)
    comprehensive = add_family_info(comprehensive, set5_df)
    comprehensive = enrich_language_names_from_sil(comprehensive, cldr_cache_dir)
    comprehensive = add_identifier_context(comprehensive)
    comprehensive = add_service_language_codes(comprehensive)

    # Apply the same SIL name map to the long-form script table. parse_cldr_json()
    # emits cldr_long_df before enrichment runs, so the same ~215 codes still have
    # language_name=code there. Reuse the already-cached SIL table rather than
    # re-downloading it.
    _sil_map = _load_sil_names(cldr_cache_dir)
    _sil_map.update(_SIL_NAME_OVERRIDES)
    _long_missing = cldr_long_df['language_name'] == cldr_long_df['language_code']
    if _long_missing.any():
        cldr_long_df = cldr_long_df.copy()
        cldr_long_df.loc[_long_missing, 'language_name'] = (
            cldr_long_df.loc[_long_missing, 'language_code'].map(
                lambda c: _sil_map.get(c, c)
            )
        )
        console.print(f"  SIL name enrichment (long form): "
                      f"{_long_missing.sum()} rows / "
                      f"{cldr_long_df.loc[_long_missing, 'language_code'].nunique()} codes fixed")

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
    name_col = 'language_name'
    n_named = (c[name_col] != c['language_code']).sum()
    console.print(f"  Total language codes:         {len(c)}")
    console.print(f"  With English name:            {n_named} / {len(c)}")
    console.print(f"  With family assigned:         {(c['iso639_5_family']!='').sum()} / {len(c)}")
    console.print(f"  ISO 639-1 (2-letter):         {c['in_iso639_1'].sum()}")
    console.print(f"  ISO 639-2 (individual):       {c['in_iso639_2'].sum()}")
    console.print(f"  Wikimedia (active Wikipedia): {c['in_wikimedia'].sum()}")
    console.print(f"  Modern languages:             {c['modern_language'].sum()}")
    console.print(f"  Ancient/extinct/constructed:  {(~c['modern_language']).sum()}")
    console.print(f"  RTL languages:                {(c['directionality']=='rtl').sum()}")
    console.print(f"\n  Top 10 families by code count:")
    top = (c[c['family_name'] != ''].groupby('family_name')
           .size().sort_values(ascending=False).head(10))
    for name, n in top.items():
        console.print(f"    {name:<44} {n}")

    console.print("\n[bold green]Sources and rationale:[/bold green]")
    console.print("  CLDR 48.2 (JSON): script data, directionality, modern-use flag, official status, family hierarchy")
    console.print("  LOC:              ISO 639-1/2 code standardisation (Kamusella 2012 critique noted)")
    console.print("  Wikimedia:        proxy for digital community presence beyond ISO")
    console.print("  CLDR v45 suppl.:  23 ISO 639-3 languages dropped from CLDR 48.2 for contributor reasons only")
    console.print("  SIL ISO 639-3:    English reference names for ~190 CLDR codes absent from CLDR en/languages.json")

    return comprehensive, cldr_long_df, set5_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build comprehensive language codes: CLDR (JSON) + LOC + Wikimedia"
    )
    parser.add_argument('--cldr-version', default='48.2.0',
        help='CLDR JSON version to fetch from npm (default: 48.2.0)')
    parser.add_argument('--cldr-cache-dir', default=None,
        help='Directory to cache downloaded CLDR npm packages '
             '(default: <output-dir>/cldr-cache)')
    parser.add_argument('--loc-file',
        default=None,
        help='LOC ISO 639-2 URL or local HTML file '
             '(default: https://www.loc.gov/standards/iso639-2/php/code_list.php)')
    parser.add_argument('--wikimedia-file',
        default=None,
        help='Wikimedia language codes URL or local CSV '
             '(default: https://meta.wikimedia.org/wiki/Template:List_of_language_names_ordered_by_code)')
    parser.add_argument('--output-dir',
        default=None,
        help='Output directory (default: metadata_files/ under DATA_DIR)')
    args = parser.parse_args()
    main(cldr_version=args.cldr_version,
         cldr_cache_dir=args.cldr_cache_dir,
         loc_source=args.loc_file,
         wikimedia_source=args.wikimedia_file,
         output_dir=args.output_dir)