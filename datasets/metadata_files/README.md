# `datasets/metadata_files/`

Project-level metadata about the languages, scripts, families, and service-support manifests the translation pipeline operates over. **Nothing in this folder is per-term translation data** — those live under `datasets/translated_terms/{term_slug}/`. This folder is the *registry layer* that defines what counts as a language for the project, what each row's institutional provenance is, and what external services can do with each code.

## Quick map

| Tier | Files |
| --- | --- |
| Primary dataset | `language_codes_comprehensive.csv` |
| Derived from primary | `language_scripts_long.csv`, `iso_639_set5.csv` |
| Per-pair editorial decisions | `family_reconciliation.csv`, `language_family_assignments.json` |
| Service-support snapshots | `service_language_code_support.csv`, `service_language_code_support.backup_*.csv` |
| Validation results | `bcp47_spotcheck_results.csv` |
| Cached upstream source data | `cldr-cache/`, `glottolog-cache/`, `iso_639_choices_directionality_wikimedia.csv` |
| Analysis artifacts | `source_overlap_venn.csv` |
| Legacy / superseded | `iso_639_choices.csv`, `iso_639_extended_choices.csv` |

The build script is `scripts/experiment/generate_language_codes.py`; see [scripts/experiment/README.md](../../scripts/experiment/README.md) for the orchestration, and [notebooks/01_language_exploration.ipynb](../../notebooks/01_language_exploration.ipynb) for analytical context on every column documented here.

---

## `language_codes_comprehensive.csv` — the primary dataset (881 rows × 39 columns)

The one row per language code that everything else in this folder either derives from or annotates. Built by merging five registries (CLDR 48.2, LOC ISO 639-2, Wikimedia, CLDR v45 supplement, ISO 639-5 family hierarchy) and enriched with SIL names, Glottolog family reconciliation, BCP 47 canonicalization, and Google service support manifests. The `language_code` column is the project's stable row identifier; it never changes between regenerations and is what every other notebook joins on.

### Identity and naming (3 columns)

| Column | Type | Description |
| --- | --- | --- |
| `language_code` | str | **The project's stable identifier.** Matches the upstream registry's code (e.g. `en`, `bat-smg`, `zh-min-nan`, `nan` for Min Nan Chinese). Never normalised, never overwritten — refactors that change identifier semantics produce a *parallel* column rather than mutating this one. |
| `language_name` | str | English display name. Sourced from CLDR's `en/languages.json` for the ~693 codes CLDR covers; from SIL ISO 639-3 for the ~190 codes CLDR has no entry for; from `_SIL_NAME_OVERRIDES` for two edge cases (`kro` = "Kru languages", `tokipona` = "Toki Pona"). |
| `local_name` | str | The language's name in its own script (autonym), e.g. `日本語` for Japanese, `Deutsch` for German. From CLDR's `<code>/languages.json` where available. |

### ISO 639 standards membership (6 columns)

| Column | Type | Description |
| --- | --- | --- |
| `iso639_1` | str | Two-letter ISO 639-1 code if one exists (e.g. `en`, `de`). Empty for languages without 2-letter codes (most of the long tail). |
| `iso639_2_t` | str | Three-letter ISO 639-2 *terminological* code — the linguistically-derived form (e.g. `deu` for German). |
| `iso639_2_b` | str | Three-letter ISO 639-2 *bibliographic* code — the library-cataloging form (e.g. `ger` for German). Differs from `_t` for ~20 languages where library tradition and linguistic naming diverge. |
| `in_iso639_1` | bool | True if this code has an ISO 639-1 entry. |
| `in_iso639_2` | bool | True if this code is in LOC's ISO 639-2 list of *individual* languages (excludes collection codes). |
| `is_group_iso639_2` | bool | True if this code is a *group/collection* code in ISO 639-2 (e.g. `sla` for Slavic languages). The pipeline retains these for family reconciliation but they aren't translation targets. |

### Scripts and writing systems (5 columns)

| Column | Type | Description |
| --- | --- | --- |
| `scripts` | str | Pipe-separated list of script *names* used by this language (e.g. `Arabic\|Latin`). |
| `script_codes` | str | Pipe-separated list of ISO 15924 script *codes* (e.g. `Arab\|Latn`). |
| `primary_script` | str | The most-used modern script name. |
| `primary_script_code` | str | ISO 15924 code for the primary script. |
| `n_scripts` | int | Number of distinct scripts in `scripts`. |

### Directionality (2 columns)

| Column | Type | Description |
| --- | --- | --- |
| `directionality` | `'rtl'` or `'ltr'` | The pipeline's authoritative directionality, derived from `primary_script_code` against the ISO 15924 RTL set. Used by all downstream rendering. |
| `directionality_wikimedia` | `'rtl'`, `'ltr'`, or `''` | The Wikimedia community's recorded value where available. Preserved separately so digraphia cases (a language with both RTL and LTR community usage, like `uz_AF`) remain visible. |

### Vitality and status (2 columns)

| Column | Type | Description |
| --- | --- | --- |
| `modern_language` | bool | False for the 25 languages CLDR 48 flags as non-modern (ancient/extinct: Avestan, Akkadian, Ancient Egyptian, Linear A, etc.). The flag tracks *script vitality*, not language vitality — all 25 are retained for translation. |
| `is_official` | bool | True if the language is official in at least one country per CLDR `territoryInfo`. |

### Provenance and accumulation (5 columns)

| Column | Type | Description |
| --- | --- | --- |
| `sources` | str | Pipe-separated list of which registries attest this code, in canonical order: `cldr\|iso639_1\|iso639_2\|wikimedia\|cldr_v45`. The single most informative provenance column — answers "which institutional perspectives carry this language?" |
| `cldr_version` | str | `'48.2.0'` for codes from the live CLDR fetch, `'45'` for the 23 CLDR v45 supplement codes (dropped between CLDR 45 and 48 for administrative reasons but retained here), `''` for Wikimedia-only or LOC-only rows. |
| `in_wikimedia` | bool | True if this code currently has an active Wikipedia project per the live Wikimedia language template. |
| `coding_dh_date_added` | date | When this row first appeared in any regeneration of the pipeline. Bootstrapped from git history for legacy rows; set to today for newly-attested codes on each future regeneration. See notebook 01's "Date provenance" cell for the methodological rationale. |
| `coding_dh_date_last_seen` | date | When this row was last confirmed present in current upstream sources. A row whose `last_seen` predates today indicates that the registries stopped attesting the code on that date — the row is *retained* (not dropped) with all its previous metadata frozen. This generalises the CLDR v45 supplement logic to every source. |

### Family classification (5 columns)

| Column | Type | Description |
| --- | --- | --- |
| `iso639_5_direct` | str | Direct ISO 639-5 group code from CLDR `languageGroups.json` (e.g. `gem` for Germanic). Empty for codes outside the CLDR family hierarchy. |
| `iso639_5_family` | str | Top-level ISO 639-5 family code, traversed up to depth 0 (e.g. `ine` for Indo-European). |
| `family_name` | str | English name of the top-level family per ISO 639-5 (e.g. `Indo-European languages`). Three-tier assignment: manual table (292 codes), `language_family_assignments.json` JSON fallback (586 codes), ISO 639-5 direct (2 codes), covering all 881 rows. |
| `subfamily_name` | str | English name of the direct group when the row sits at depth ≥ 1 (e.g. `Slavic languages` for Russian, whose family is `Indo-European languages`). |
| `family_name_reconciled` | str | **Glottolog 5.3 cross-validated family.** Equals `family_name` when ISO 639-5 and Glottolog agree (586 codes); equals Glottolog's classification where ISO 639-5 uses geographic groupings (`Caucasian`, `North American Indian`) or abandoned/contested macrofamilies (`Niger-Kordofanian`, `Altaic`) — 295 reclassifications. Per-pair decisions live in `family_reconciliation.csv`. `get_language_family()` in `scripts/utils.py` prefers this column over `family_name`, so downstream notebooks pick up the reconciled classification automatically. |

### BCP 47 interoperability (7 columns)

These columns add an interoperability layer for web/API contexts without mutating `language_code`. The whole layer was integrated late (see nb01 §1.1's "Late integration" footnote); canonicalization delegates to the `langcodes` library, with seven Wikimedia legacy tags absent from the 2021-08-06 IANA snapshot handled by `BCP47_GRANDFATHERED_FALLBACKS` in the build script.

| Column | Type | Description |
| --- | --- | --- |
| `project_language_code` | str | A copy of `language_code` preserved under a name that makes the "stable project identifier" semantics explicit when consumers also see `bcp47_tag`. |
| `wikimedia_code` | str | The exact Wikimedia code where this row has Wikimedia attestation (i.e. equals `language_code` if `in_wikimedia` is True; empty otherwise). Useful when downstream code wants to query Wikipedia explicitly. |
| `bcp47_tag` | str | IANA-canonical BCP 47 tag for this code (e.g. `sgs` for `bat-smg`, `he` for `iw`, `fil` for `tl`, `sr-ME` for `cnr`). For 866 of 881 rows this just mirrors `language_code` with hyphen/case normalisation; for 15 rows it carries IANA's Preferred-Value rewrite or a grandfathered-fallback mapping. |
| `bcp47_source` | str | Where `bcp47_tag` came from: `iso639_1` / `iso639_2_t` / `language_code` (mechanical normalization), or `grandfathered_fallback` (the seven Wikimedia tags absent from langcodes' IANA snapshot). |
| `bcp47_note` | str | Free-text rationale, primarily for grandfathered fallbacks (e.g. *"Wikimedia/legacy code bat-smg not in language_data IANA snapshot; mapped manually"*). Empty for mechanical normalizations. |
| `bcp47_is_project_code` | bool | True when `bcp47_tag` is just a hyphen/case rewrite of `language_code`. False when IANA Preferred-Value or a grandfathered fallback actually changed the primary subtag — these are the rows downstream code should flag if it cares about the protocol-level identifier diverging from the project-level one. |
| `service_code_candidates` | str | Pipe-separated list of identifiers to test against external service support lists. Includes `language_code`, `bcp47_tag`, `wikimedia_code`, the ISO 639-1/2 codes, and legacy aliases from IANA's full Replacements table (e.g. `iw` for Hebrew rows because Google's NMT manifest still keys on `iw`). Lets a service match succeed when the service uses the deprecated form. |

### Service support (4 columns)

The pipeline carries support flags for Google Translate's two endpoints, populated by joining each row's `service_code_candidates` against `service_language_code_support.csv`. Adding a new service is a matter of appending rows to that manifest with the right `service` value; the build script automatically emits `<service>_code` and `<service>_supported` columns for any service it finds.

| Column | Type | Description |
| --- | --- | --- |
| `google_nmt_code` | str | The exact code Google's standard NMT endpoint accepts for this row (e.g. `iw` for Hebrew, even though `bcp47_tag` is `he`), or empty if unsupported. |
| `google_nmt_supported` | bool | True if this row has any matched candidate in Google NMT's supported-languages manifest. Currently ~184/881 (21%). |
| `google_translation_llm_code` | str | The exact code Google's Translation LLM endpoint accepts. The LLM tier has narrower coverage than NMT. |
| `google_translation_llm_supported` | bool | True if this row has any matched candidate in Translation LLM's manifest. Currently ~87/881 (10%). The NMT-vs-LLM gap is a paper finding about the long tail of unsupported scholarly languages. |

---

## Derived datasets

### `language_scripts_long.csv` — one row per (language × script) pair

Long-form version of the `scripts` column from the primary CSV. Used for script-switching analysis and any visualisation that wants to plot each language–script pair as a separate observation. Columns: `language_name`, `language_code`, `modern_language`, `is_official`, `script_name`, `script_code`, `script_modern`, `is_rtl`. Regenerated alongside the primary CSV by `generate_language_codes.py`.

### `iso_639_set5.csv` — ISO 639-5 family hierarchy

CLDR's `languageGroups.json` flattened into a parent-pointer table. Columns: `iso639_5` (the group code, e.g. `gem`), `parent_code` (the immediate parent in the hierarchy), `family_name` (English name), `depth` (distance from the root). 113 rows covering 27 top-level families. Consumed by `add_family_info()` in the build script to populate `iso639_5_family` and `family_name` on the primary CSV.

---

## Per-pair editorial decisions

### `family_reconciliation.csv` — Glottolog cross-validation decisions

One row per (ISO 639-5 family × Glottolog family) disagreement pair, with an explicit decision recorded. Columns: `family_name_iso`, `family_name_glottolog`, `n_languages` (count of pipeline languages affected by this pair), `sample_languages` (examples), `choice` (`'glottolog'` or `'iso'`), `category` (`'abandoned-macrofamily'`, `'geographic→linguistic'`, `'metadata'`, `'granularity'`, etc.), `chosen_family_name` (what `family_name_reconciled` should be for languages matching this pair), `rationale` (free text). Authored in notebook 01 §1.3; applied by `add_family_reconciliation()` in the build script. **If `generate_language_codes.py` warns about uncovered disagreement pairs after a regen**, open `html_files/family_reconciliation_reviewer.html` to add decisions for them, then re-run.

### `language_family_assignments.json` — JSON fallback for family assignment

The middle tier of the three-tier family assignment used by `add_family_info()`: manual table (292 codes) → this JSON (586 codes) → ISO 639-5 direct (2 codes). Each entry records a `language_code`, a `family_name`, and an `iso639_5` code where applicable, plus a brief rationale. Authored once by `generate_family_assignments.py` and committed; downstream regenerations read it as-is. To revise assignments, edit the `ASSIGNMENTS` dict in that script and re-run it before re-running `generate_language_codes.py`.

---

## Service-support snapshots

### `service_language_code_support.csv` — Google Translate support manifest

The current snapshot of which language codes Google's two Translation API endpoints accept. Columns: `service` (`google_nmt` or `google_translation_llm`), `service_code` (the exact identifier Google's API uses), `language_name`, `support_level` (`'official'` or `'experimental'`), `model` (`'nmt'` or `'translation_llm'`), `notes`, `snapshot_date` (ISO date of the API call), `snapshot_source` (`'translate_v3.get_supported_languages'` for refreshed rows, `'preserved_from_previous'` when the regional LLM endpoint refused and the previous LLM rows were kept). Refresh via `python -m scripts.experiment.refresh_service_support`. The script is dated-snapshot-aware: every refresh writes the date into the rows it touches, so old analyses can be checked against the snapshot they ran against.

### `service_language_code_support.backup_<YYYY-MM-DD>.csv` — automatic backups

Created by `refresh_service_support.py` before it overwrites the live manifest, so a regen can always be rolled back. Safe to delete once you no longer need the old snapshot; not consumed by any downstream code.

---

## Validation results

### `bcp47_spotcheck_results.csv` — does BCP 47 normalization change MT output?

Per-row results from `scripts/experiment/spotcheck_bcp47_codes.py`. For each row in the primary CSV where `bcp47_tag` substantively differs from `language_code`, the script queries Google Translate's v2 API under *both* forms and reports whether the service returns identical, divergent, or asymmetric (one-form-fails) translations. Columns: `language_code`, `language_name`, `bcp47_tag`, `kind` (`'iana_preferred_value'` or `'grandfathered'`), `bcp47_source`, `project_translation`, `project_error`, `bcp47_translation`, `bcp47_error`, `agree` (bool), `outcome` (`'agree'` / `'diverge'` / `'project_only'` / `'bcp47_only'` / `'both_failed'`). The headline finding (zero divergences across the cases Google accepts both forms) validates the pipeline's choice to query under `language_code`. See nb01 §1.1's BCP 47 validation cell for the full discussion.

---

## Cached upstream source data

### `cldr-cache/`

The `cldr-core` and `cldr-localenames-full` npm packages, plus the SIL ISO 639-3 reference table, cached after the first build run. Subsequent regenerations are fully offline once this is populated. Not committed to git; rebuilt automatically by `generate_language_codes.py` on first run.

### `glottolog-cache/`

Glottolog 5.3 languoid table (`languoid.csv`) and a `README.txt` with citation info. Read by `add_family_reconciliation()` to compute `family_name_reconciled`. The Glottolog data is CC-BY-4.0 — cite as: *Hammarström, Harald & Robert Forkel & Martin Haspelmath & Sebastian Bank. 2026. Glottolog 5.3. Leipzig: Max Planck Institute for Evolutionary Anthropology.* (See `glottolog-cache/README.txt` for the canonical citation.)

### `iso_639_choices_directionality_wikimedia.csv` — Wikimedia snapshot

A frozen snapshot of the Wikimedia language list with manual annotations for documented code-reuse collisions (e.g. `als,Alemannic,...,en:ISO 639-3: gsw (als is en:Tosk Albanian)`). This is **not** the source the build script reads from by default — `generate_language_codes.py` fetches the Wikimedia template live from `meta.wikimedia.org`. The local snapshot exists for reproducibility (pin the build with `--wikimedia-file path/to/this/csv` to get deterministic regenerations) and as documentation of which collisions have been audited. The local snapshot can lag the live list: the 2026-06 regen surfaced `pnb` (Punjabi Western) and `crn` (Wikimedia legacy for Montenegrin) as present in the live data but missing from this snapshot.

---

## Analysis artifacts

### `source_overlap_venn.csv`

Pre-computed counts for the three-source overlap Venn diagram in notebook 01 §1.1. Two columns: `combination` (e.g. `CLDR + ISO 639-2 + Wikimedia`) and `count`. Written by the notebook for downstream RAW Graphs use; safe to regenerate by re-running that cell.

---

## Legacy / superseded files

### `iso_639_choices.csv` and `iso_639_extended_choices.csv`

Older two-column files (`language`, `name`) used before the comprehensive build pipeline existed. Kept for historical reference only — superseded by `language_codes_comprehensive.csv` for every downstream use. Safe to delete if you don't need the historical record.

---

## How to regenerate

```bash
# Primary build (also writes language_scripts_long.csv and iso_639_set5.csv)
python -m scripts.experiment.generate_language_codes

# Refresh service support manifest from Google's API
python -m scripts.experiment.refresh_service_support

# Validate the BCP 47 layer against Google Translate
python -m scripts.experiment.spotcheck_bcp47_codes \
    --csv-out datasets/metadata_files/bcp47_spotcheck_results.csv

# Quick diagnostic on identifier columns
python scripts/experiment/audit_language_identifiers.py \
    datasets/metadata_files/language_codes_comprehensive.csv
```

The primary build prints a "Recommended next steps" panel at the end pointing at the family-reconciliation HTML reviewer and notebook 01. See [scripts/experiment/README.md](../../scripts/experiment/README.md) for the full script reference.
