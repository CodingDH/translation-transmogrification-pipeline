"""
parse_cldr_json.py
==================
Fetches and parses Unicode CLDR data from the canonical npm packages, used by generate_language_codes.py to build the comprehensive language codes dataset.

Two npm packages are downloaded and cached locally (no HTML scraping):
1. cldr-core (~200 KB): supplemental data — languageData, scriptMetadata, territoryInfo,languageGroups, aliases
2. cldr-localenames-full (~2 MB): English display names for languages, scripts, and territories

Both are pinned to a specific CLDR version (default: 48.2.0), but could work with any version that has the same JSON structure. The 'cldr_version' provenance column in the output CSV records which version was used. To use a different version, just delete the cached packages and re-run, specifying the new version.

Public functions
----------------
parse_cldr_json(cldr_version, cache_dir)
    → (lang_df, long_df)
    One row per language code / (language × script) pair.
    Columns: language_code, language_name, directionality, scripts,
    modern_language, is_official, cldr_version, …

parse_cldr_family_groups(cldr_version, cache_dir)
    → family_df
    CLDR languageGroups hierarchy in the same schema expected by
    add_family_info() in generate_language_codes.py.
    Columns: iso639_5, parent_code, family_name, depth

parse_cldr_aliases(cldr_version, cache_dir)
    → aliases_df
    Deprecated/legacy language code mappings (e.g. 'iw' → 'he').
    Columns: code, replacement, reason

Design notes
------------
- modern_language is inferred from scriptMetadata idUsage:
  RECOMMENDED or LIMITED_USE → modern; all-EXCLUSION → not modern.
- Official-language status comes from territoryInfo _officialStatus
  ('official' or 'de_facto_official' in any territory).
- Directionality uses the primary script's rtl flag in scriptMetadata,
  cross-checked against RTL_SCRIPT_CODES.

Citing the data
---------------
In the methods section cite: Unicode CLDR version 48.2.0, https://github.com/unicode-org/cldr-json. The 'cldr_version' provenance column in the output CSV records which
version was used.
"""

import json
import os
import tarfile
from io import BytesIO
from urllib.request import urlopen

import pandas as pd

# ── Constants reused from generate_language_codes.py ─────────────────────────
RTL_SCRIPT_CODES = {
    'Arab', 'Hebr', 'Thaa', 'Syrc', 'Nkoo', 'Adlm', 'Rohg', 'Tfng',
    'Mend', 'Yezi', 'Nbat', 'Palm', 'Hatr', 'Narb', 'Sarb', 'Armi',
    'Avst', 'Hung', 'Lydi', 'Ital', 'Phnx', 'Prti', 'Phli', 'Khar',
    'Cprt', 'Linb', 'Sogd',
}


# ── npm tarball fetcher ──────────────────────────────────────────────────────

def _fetch_npm_package(package: str, version: str, cache_dir: str) -> str:
    """
    Download and extract a CLDR JSON package from the npm registry if not already cached. Returns the path to the extracted package directory.
    
    Parameters
    ----------
    package : str
		The npm package name (e.g. 'cldr-core').
    version : str
		The CLDR version to fetch (e.g. '48.2.0').
	cache_dir : str
		Local directory to cache downloaded packages. Each package is stored under a subdirectory named '{package}-{version}' (e.g. 'cldr-core-48.2.0'). If the subdirectory already exists and is non-empty, it is reused without re-downloading.
        
    Returns
	-------
	str
		The path to the extracted package directory containing the JSON files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    pkg_dir = os.path.join(cache_dir, f'{package}-{version}')
    if os.path.isdir(pkg_dir) and os.listdir(pkg_dir):
        return pkg_dir

    url = f'https://registry.npmjs.org/{package}/-/{package}-{version}.tgz'
    print(f'  Fetching {package} {version} from npm...')
    with urlopen(url, timeout=60) as resp:
        data = resp.read()
    print(f'  Got {len(data)/1024:.0f} KB; extracting to {pkg_dir}')

    os.makedirs(pkg_dir, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(data), mode='r:gz') as tf:
        tf.extractall(pkg_dir)

    # npm tarballs put everything under 'package/', flatten that
    inner = os.path.join(pkg_dir, 'package')
    if os.path.isdir(inner):
        for item in os.listdir(inner):
            os.rename(os.path.join(inner, item), os.path.join(pkg_dir, item))
        os.rmdir(inner)

    return pkg_dir


# ── Main parser ──────────────────────────────────────────────────────────────

def parse_cldr_json(cldr_version: str = '48.2.0', cache_dir: str = './cldr-cache') -> tuple:
    """
    Build (lang_df, long_df) from CLDR JSON.
    
    Parameters
	----------
    cldr_version : str
		The CLDR version to fetch and parse (e.g. '48.2.0'). The 'cldr_version' provenance column in the output CSV records which version was used. To use a different version, just delete the cached packages and re-run, specifying the new version.
	cache_dir : str
		Local directory to cache downloaded packages. See _fetch_npm_package() for details.

    Returns
    -------
    lang_df : DataFrame, one row per language code. Columns include
        language_name, language_code, modern_language, is_official,
        scripts, script_codes, primary_script, primary_script_code,
        directionality, n_scripts, cldr_version.
    long_df : DataFrame, one row per (language, script) pair. Columns:
        language_name, language_code, modern_language, is_official,
        script_name, script_code, script_modern, is_rtl.
    """
    core_dir = _fetch_npm_package('cldr-core', cldr_version, cache_dir)
    names_dir = _fetch_npm_package('cldr-localenames-full', cldr_version, cache_dir)

    # ── Load the five JSON files we need ─────────────────────────────────────
    def _load(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    language_data = _load(os.path.join(core_dir, 'supplemental', 'languageData.json'))
    script_meta   = _load(os.path.join(core_dir, 'scriptMetadata.json'))
    territory_info = _load(os.path.join(core_dir, 'supplemental', 'territoryInfo.json'))
    en_names_doc  = _load(os.path.join(names_dir, 'main', 'en', 'languages.json'))

    ld_entries = language_data['supplemental']['languageData']
    sm = script_meta['scriptMetadata']
    ti = territory_info['supplemental']['territoryInfo']
    en_names = en_names_doc['main']['en']['localeDisplayNames']['languages']

    # ── 1. Build per-language script lists, primary then secondary ───────────
    # CLDR splits languages into a primary entry (e.g. 'uz' = Uzbek) and an
    # optional -alt-secondary entry listing additional historical scripts.
    # We merge them, marking primary scripts as 'modern' for the language and
    # secondary scripts as additional. Within scripts, RECOMMENDED and
    # LIMITED_USE in scriptMetadata indicate currently-used scripts;
    # EXCLUSION indicates ancient/historical.
    language_scripts = {}   # code -> [(script, is_primary), ...]
    for key, entry in ld_entries.items():
        if '-alt-' in key:
            base, alt = key.split('-alt-', 1)
            is_primary = (alt != 'secondary')
        else:
            base, is_primary = key, True
        for script in entry.get('_scripts', []):
            language_scripts.setdefault(base, []).append((script, is_primary))

    # ── 2. Compute official-language status from territoryInfo ───────────────
    # 'official' = nationally official (e.g. French in France)
    # 'de_facto_official' = effectively national (e.g. English in US)
    # 'official_regional' = official in a sub-national region only (e.g. Welsh
    # in Wales). We exclude regional-only to match CLDR HTML's stricter flag.
    official_langs = set()
    for terr, info in ti.items():
        for loc, lp in info.get('languagePopulation', {}).items():
            status = lp.get('_officialStatus', '')
            if status in ('official', 'de_facto_official'):
                # Strip script/region suffixes like 'zh_Hans' or 'en_US'
                base = loc.split('_')[0].split('-')[0].lower()
                official_langs.add(base)

    # ── 3. Build the long-form (one row per (language, script) pair) ─────────
    long_records = []
    for code, script_list in language_scripts.items():
        name = en_names.get(code, code)
        is_official = code in official_langs

        # 'modern_language' for the whole language: true if at least one
        # script has idUsage RECOMMENDED or LIMITED_USE. This is the closest
        # JSON analogue to the HTML chart's modern/O column.
        modern_lang = any(
            sm.get(s, {}).get('idUsage') in ('RECOMMENDED', 'LIMITED_USE')
            for s, _ in script_list
        )

        for script, is_primary in script_list:
            script_info = sm.get(script, {})
            id_usage = script_info.get('idUsage', 'UNKNOWN')
            script_modern = (id_usage in ('RECOMMENDED', 'LIMITED_USE')) and is_primary
            long_records.append({
                'language_name':   name,
                'language_code':   code,
                'modern_language': modern_lang,
                'is_official':     is_official,
                'script_name':     script,         # CLDR uses 4-letter codes
                'script_code':     script,
                'script_modern':   script_modern,
                'is_rtl':          script_info.get('rtl') == 'YES'
                                   or script in RTL_SCRIPT_CODES,
            })

    long_df = pd.DataFrame(long_records)

    # ── 4. Aggregate to one row per language ─────────────────────────────────
    def agg_language(grp):
        # Primary script: the first primary-entry script in CLDR's languageData
        # for this language. (-alt-secondary entries come after primary ones,
        # so groupby preserves that order; the first row of each group is the
        # primary script.) Use that script — not any() across all scripts —
        # to compute directionality, so languages like Turkish that have
        # Latin primary + Arabic secondary don't get marked RTL incorrectly.
        pidx = grp.index[0]
        primary_script = grp.loc[pidx, 'script_code']
        primary_is_rtl = grp.loc[pidx, 'is_rtl']
        return pd.Series({
            'language_name':       grp['language_name'].iloc[0],
            'modern_language':     bool(grp['modern_language'].iloc[0]),
            'is_official':         bool(grp['is_official'].iloc[0]),
            'scripts':             '|'.join(grp['script_name'].tolist()),
            'script_codes':        '|'.join(grp['script_code'].tolist()),
            'primary_script':      primary_script,
            'primary_script_code': primary_script,
            'directionality':      'rtl' if primary_is_rtl else 'ltr',
            'n_scripts':           len(grp),
        })

    lang_df = (long_df.groupby('language_code', as_index=False)
               .apply(agg_language, include_groups=False)
               .reset_index(drop=True))

    # Provenance: which CLDR version produced this row
    lang_df['cldr_version'] = cldr_version

    print(f'  Parsed CLDR {cldr_version}: {len(lang_df)} languages, '
          f'{(lang_df["directionality"]=="rtl").sum()} RTL, '
          f'{lang_df["modern_language"].sum()} modern, '
          f'{lang_df["is_official"].sum()} official somewhere')

    return lang_df, long_df


# ── Family hierarchy from CLDR languageGroups (replaces ISO 639-5 scrape) ───

def _parse_cldr_language_groups(cldr_version: str = '48.2.0', cache_dir: str = './cldr-cache') -> pd.DataFrame:
    """
    Build a DataFrame from CLDR's languageGroups.json. This replaces the
    Wikipedia-scraped ISO 639-5 table because CLDR maintains its own
    machine-readable family hierarchy that uses ISO 639-5 codes for
    top-level families and adds finer-grained subdivisions.

    The JSON format is: {group_code: 'space separated list of child codes'}.
    Children can be individual languages OR other groups (so the tree is
    multi-level and we need to compute top-level family by walking up).
    
    Parameters
	----------
    cldr_version : str
		The CLDR version to fetch and parse (e.g. '48.2.0'). The 'cldr_version' provenance column in the output CSV records which version was used. To use a different version, just delete the cached packages and re-run, specifying the new version.
	cache_dir : str
		Local directory to cache downloaded packages. See _fetch_npm_package() for details.

    Returns
    -------
    DataFrame with columns:
        group_code      the ISO 639-5 / CLDR family code (e.g. 'gem')
        member_codes    pipe-separated list of immediate children
        parent_code     this group's parent in the hierarchy, if any
        is_top_level    True if this group has no parent (top family)
    """
    core_dir = _fetch_npm_package('cldr-core', cldr_version, cache_dir)
    lg = (json.load(open(os.path.join(core_dir, 'supplemental',
                                      'languageGroups.json')))
          ['supplemental']['languageGroups'])

    # Build parent map: for each child, find its parent group.
    # CLDR's hierarchy has a single synthetic root, 'mul' (= ISO 639-2 code
    # for 'multiple languages'), whose children are the real top-level
    # families. We treat 'mul' as the root sentinel, not a real family.
    parent = {}
    for group, members in lg.items():
        for m in members.split():
            # If a child is itself a group, parent[child] = group
            if m in lg:
                parent[m] = group

    top_level_families = set(lg.get('mul', '').split())

    records = []
    for group, members in lg.items():
        if group == 'mul':
            continue   # skip the synthetic root
        records.append({
            'group_code':   group,
            'member_codes': '|'.join(members.split()),
            'parent_code':  parent.get(group, ''),
            'is_top_level': group in top_level_families,
        })
    return pd.DataFrame(records).sort_values('group_code').reset_index(drop=True)


# English names for ISO 639-5 group codes. CLDR's languages.json only carries
# names for codes that double as individual-language codes (e.g. 'cr', 'sw').
# The group codes (e.g. 'gem', 'ine') need a standalone lookup. These codes
# are stable ISO 639-5 identifiers, so a hardcoded dict is appropriate.
_ISO639_5_NAMES = {
    # top-level families
    'aav': 'Austro-Asiatic languages',
    'afa': 'Afro-Asiatic languages',
    'art': 'Artificial languages',
    'aus': 'Australian languages',
    'cai': 'Central American Indian languages',
    'cau': 'Caucasian languages',
    'crp': 'Creoles and pidgins',
    'cpe': 'Creoles and pidgins, English-based',
    'dra': 'Dravidian languages',
    'esx': 'Eskimo-Aleut languages',
    'euq': 'Basque languages',
    'hmx': 'Hmong-Mien languages',
    'ine': 'Indo-European languages',
    'jpx': 'Japonic languages',
    'khi': 'Khoisan languages',
    'map': 'Austronesian languages',
    'nai': 'North American Indian languages',
    'nic': 'Niger-Kordofanian languages',
    'paa': 'Papuan languages',
    'sai': 'South American Indian languages',
    'sgn': 'Sign languages',
    'sit': 'Sino-Tibetan languages',
    'ssa': 'Nilo-Saharan languages',
    'tai': 'Tai-Kadai languages',
    'tup': 'Tupian languages',
    'tut': 'Altaic languages',
    'urj': 'Uralic languages',
    'ypk': 'Yupik languages',
    # depth-1 sub-groups
    'alv': 'Atlantic-Volta languages',
    'aqa': 'Alacalufan languages',
    'aql': 'Algonquian-Mosan languages',
    'awd': 'Arawakan languages',
    'azc': 'Uto-Aztecan languages',
    'bat': 'Baltic languages',
    'ber': 'Berber languages',
    'bnt': 'Bantu languages',
    'btk': 'Batak languages',
    'cba': 'Chibchan languages',
    'ccn': 'North Caucasian languages',
    'ccs': 'South Caucasian languages',
    'cdc': 'Chadic languages',
    'cdd': 'Caddoan languages',
    'cel': 'Celtic languages',
    'cmc': 'Chamic languages',
    'cpf': 'Creoles and pidgins, French-based',
    'cpp': 'Creoles and pidgins, Portuguese-based',
    'csu': 'Central Sudanic languages',
    'cus': 'Cushitic languages',
    'day': 'Land Dayak languages',
    'dmn': 'Mande languages',
    'egx': 'Egyptian languages',
    'fiu': 'Finno-Ugrian languages',
    'fox': 'Formosan languages',
    'gem': 'Germanic languages',
    'gme': 'East Germanic languages',
    'gmq': 'North Germanic languages',
    'gmw': 'West Germanic languages',
    'grk': 'Greek languages',
    'hok': 'Hokan languages',
    'iir': 'Indo-Iranian languages',
    'inc': 'Indo-Aryan languages',
    'ira': 'Iranian languages',
    'iro': 'Iroquoian languages',
    'itc': 'Italic languages',
    'ijo': 'Ijo languages',
    'kar': 'Karen languages',
    'kdo': 'Kordofanian languages',
    'kro': 'Kru languages',
    'mkh': 'Mon-Khmer languages',
    'mno': 'Manobo languages',
    'mun': 'Munda languages',
    'myn': 'Mayan languages',
    'nah': 'Nahuatl languages',
    'ngb': 'Gbaya languages',
    'ngf': 'Trans-New Guinea languages',
    'nub': 'Nubian languages',
    'omq': 'Oto-Manguean languages',
    'omv': 'Omotic languages',
    'oto': 'Otomian languages',
    'phi': 'Philippine languages',
    'plf': 'Central Malayo-Polynesian languages',
    'poz': 'Malayo-Polynesian languages',
    'pqe': 'Eastern Malayo-Polynesian languages',
    'pqw': 'Western Malayo-Polynesian languages',
    'qwe': 'Quechuan languages',
    'roa': 'Romance languages',
    'sal': 'Salishan languages',
    'sdv': 'Eastern Sudanic languages',
    'sem': 'Semitic languages',
    'sio': 'Siouan languages',
    'sla': 'Slavic languages',
    'smi': 'Sami languages',
    'son': 'Songhai languages',
    'syd': 'Samoyedic languages',
    'tbq': 'Tibeto-Burman languages',
    'trk': 'Turkic languages',
    'tuw': 'Tungus languages',
    'wak': 'Wakashan languages',
    'wen': 'Sorbian languages',
    'xgn': 'Mongolian languages',
    'xnd': 'Na-Dene languages',
    'zhx': 'Chinese languages',
    'zle': 'East Slavic languages',
    'zls': 'South Slavic languages',
    'zlw': 'West Slavic languages',
    'znd': 'Zande languages',
    # depth-2+
    'alg': 'Algonquian languages',
    'apa': 'Apache languages',
    'ath': 'Athapascan languages',
    'auf': 'Arauan languages',
    'bad': 'Banda languages',
    'bai': 'Bamileke languages',
    'cr':  'Cree',
    'lgc': 'Logographic character systems',
    'sqj': 'Albanian languages',
    'hyx': 'Armenian languages',
}


# ── Family groups: set5-compatible DataFrame from CLDR languageGroups ────────

def parse_cldr_family_groups(cldr_version: str = '48.2.0', cache_dir: str = './cldr-cache') -> pd.DataFrame:
    """
    Build a DataFrame from CLDR's languageGroups.json with the schema expected
    by add_family_info() in generate_language_codes.py.

    cldr-core is already downloaded by parse_cldr_json() so there is no extra network cost. English family names come from cldr-localenames-full's en/languages.json (also already cached), supplemented by _ISO639_5_NAMES for group codes not present in that file. Codes not resolvable from either source get an empty name (same behaviour as unmapped codes in the Wikipedia scraper).
    
    Parameters
	----------
    cldr_version : str
		The CLDR version to fetch and parse (e.g. '48.2.0'). The 'cldr_version' provenance column in the output CSV records which version was used. To use a different version, just delete the cached packages and re-run, specifying the new version.
	cache_dir : str
		Local directory to cache downloaded packages. See _fetch_npm_package() for details.

    Returns
    -------
    DataFrame with columns:
        iso639_5    ISO 639-5 / CLDR family code (e.g. 'gem', 'ine')
        parent_code parent group code, or '' for top-level families
        family_name English name from CLDR (may be '' for rare sub-groups)
        depth       0 = top-level family, 1 = sub-group, etc.
    """
    core_dir  = _fetch_npm_package('cldr-core', cldr_version, cache_dir)
    names_dir = _fetch_npm_package('cldr-localenames-full', cldr_version, cache_dir)

    lg = (json.load(open(os.path.join(core_dir, 'supplemental', 'languageGroups.json')))
          ['supplemental']['languageGroups'])
    en_names = (json.load(open(os.path.join(names_dir, 'main', 'en', 'languages.json')))
                ['main']['en']['localeDisplayNames']['languages'])

    # Build parent map (skip synthetic root 'mul')
    parent = {}
    for group, members in lg.items():
        for m in members.split():
            if m in lg:
                parent[m] = group

    top_level = set(lg.get('mul', '').split())

    # BFS to compute depth: top-level families = 0, their children = 1, ...
    from collections import deque
    depth_map: dict = {}
    queue = deque((code, 0) for code in top_level)
    while queue:
        code, d = queue.popleft()
        if code in depth_map:
            continue
        depth_map[code] = d
        for m in lg.get(code, '').split():
            if m in lg and m not in depth_map:
                queue.append((m, d + 1))

    records = []
    for group in lg:
        if group == 'mul':
            continue
        records.append({
            'iso639_5':   group,
            'parent_code': parent.get(group, ''),
            'family_name': (en_names.get(group)
                            or _ISO639_5_NAMES.get(group, '')),
            'depth':       depth_map.get(group, 0),
        })

    df = pd.DataFrame(records).sort_values(['depth', 'iso639_5']).reset_index(drop=True)
    print(f'  CLDR language groups {cldr_version}: {len(df)} groups, '
          f'{(df["depth"] == 0).sum()} top-level families, '
          f'{df["family_name"].ne("").sum()} with English names')
    return df


# ── Aliases (deprecated/legacy code mappings) ────────────────────────────────

def parse_cldr_aliases(cldr_version: str = '48.2.0', cache_dir: str = './cldr-cache') -> pd.DataFrame:
    """
    Build a DataFrame of CLDR's language aliases — codes that CLDR treats as deprecated or legacy and maps to a 'real' replacement. Useful for interpreting Wikimedia-only codes that ISO has since superseded.

    Example rows:
        mo  -> ro       (Moldovan deprecated to Romanian)
        cnr -> sr-ME    (Montenegrin ISO 639-3 mapped to Serbian-Montenegro)
        sh  -> sr-Latn  (Serbo-Croatian legacy to Serbian-Latin)
        iw  -> he       (Hebrew old code deprecated)
        
    Parameters
	----------
    cldr_version : str
		The CLDR version to fetch and parse (e.g. '48.2.0'). The 'cldr_version' provenance column in the output CSV records which version was used. To use a different version, just delete the cached packages and re-run, specifying the new version.
	cache_dir : str
		Local directory to cache downloaded packages. See _fetch_npm_package() for details.
        
    Returns
    -------
    DataFrame with columns:
		code        the deprecated/legacy code (e.g. 'iw')
		replacement the modern code that CLDR maps it to (e.g. 'he')
		reason      the reason for deprecation, if provided (e.g. 'deprecated', 'legacy', 'macrolanguage', etc.)
    """
    core_dir = _fetch_npm_package('cldr-core', cldr_version, cache_dir)
    a = (json.load(open(os.path.join(core_dir, 'supplemental', 'aliases.json')))
         ['supplemental']['metadata']['alias']['languageAlias'])
    records = [{
        'code':        code,
        'replacement': info.get('_replacement', ''),
        'reason':      info.get('_reason', ''),
    } for code, info in a.items()]
    return pd.DataFrame(records).sort_values('code').reset_index(drop=True)


if __name__ == '__main__':
    # Smoke test
    print('=== CLDR JSON parser smoke test ===')
    lang_df, long_df = parse_cldr_json(cldr_version='48.2.0',
                                       cache_dir='./cldr-cache')
    print()
    print(f'lang_df: {len(lang_df)} rows, columns = {list(lang_df.columns)}')
    print(f'long_df: {len(long_df)} rows')
    print()
    print('Sample rows (Turkish, Uzbek, Min Nan, Montenegrin, Klingon):')
    for code in ['tr', 'uz', 'nan', 'crn', 'tlh', 'eo']:
        row = lang_df[lang_df['language_code'] == code]
        if not row.empty:
            r = row.iloc[0]
            print(f'  {code:6s} {r["language_name"]:25s} '
                  f'scripts={r["scripts"]:30s} '
                  f'dir={r["directionality"]} '
                  f'modern={r["modern_language"]} '
                  f'official={r["is_official"]}')
        else:
            print(f'  {code:6s} (not in CLDR)')

    print()
    fam_df = _parse_cldr_language_groups(cldr_version='48.2.0',
                                         cache_dir='./cldr-cache')
    print(f'language groups: {len(fam_df)} rows, '
          f'{fam_df["is_top_level"].sum()} top-level families')

    print()
    al_df = parse_cldr_aliases(cldr_version='48.2.0', cache_dir='./cldr-cache')
    print(f'aliases: {len(al_df)} deprecated/legacy code mappings')
    print('Sample interesting aliases:')
    for code in ['mo', 'cnr', 'sh', 'iw', 'in', 'ji']:
        row = al_df[al_df['code'] == code]
        if not row.empty:
            r = row.iloc[0]
            print(f'  {code:6s} -> {r["replacement"]:10s} ({r["reason"]})')