# `scripts/experiment/curation/`

Curation scripts that are *not part of the standard pipeline run*. They live here, separated from the active pipeline code one directory up because they are curatorial and invoked on demand: when you want to audit the comprehensive CSV after a regeneration, refresh the Google service-support manifest from the API, validate a methodological question with a small spot-check, or rebuild a one-time data artefact.

Nothing in this folder is imported by the active pipeline. Each script is a standalone CLI tool.

## Files

### `audit_language_identifiers.py`

Lightweight diagnostic for the BCP 47 and service-support columns in `language_codes_comprehensive.csv`. Does not modify the CSV or rerun anything; reads only.

- Reports total rows and any project codes containing hyphens or underscores (e.g. `bat-smg`, `uz_AF`) — these are the rows most likely to carry interesting BCP 47 rewrites or grandfathered fallbacks.
- Lists every row where `bcp47_tag` differs from the underscore-normalized `language_code`, with the `bcp47_source` and `bcp47_note` columns side by side.
- Counts per-service support coverage from any `*_supported` columns added by `add_service_language_codes()`.

```bash
# default: audits the project's canonical comprehensive CSV
python -m scripts.experiment.curation.audit_language_identifiers

# or point at a backup / non-canonical file
python -m scripts.experiment.curation.audit_language_identifiers /path/to/some_other.csv
```

Use this after re-running `generate_language_codes.py` or refreshing the service-support manifest to confirm the columns look right before committing the regenerated CSV. CSV loading delegates to `scripts.utils.read_csv_file`, which already preserves the literal string `'nan'` (Min Nan Chinese, ISO 639-3) via `converters={'language_code': str}`; empty cells are then `.fillna("")` so audit tables show blank rather than `NaN`.

---

### `refresh_service_support.py`

Refreshes `datasets/metadata_files/service_language_code_support.csv` from Google Translate's v3 API by calling `translate_v3.get_supported_languages` for both endpoints — NMT on the `global` endpoint and Translation LLM on `us-central1`. Snapshots are stamped with `snapshot_date` and `snapshot_source` on every row, and the previous CSV is backed up to `service_language_code_support.backup_<YYYY-MM-DD>.csv` before being overwritten.

Designed to degrade gracefully: if the Translation LLM endpoint refuses or returns zero results (it is region-pinned and occasionally unreachable), the script preserves the previous LLM rows under `snapshot_source = 'preserved_from_previous'` rather than dropping them. This means coverage analyses don't silently lose data when one of the two endpoints is temporarily unavailable.

```bash
python -m scripts.experiment.curation.refresh_service_support
python -m scripts.experiment.curation.refresh_service_support --dry-run   # print diff, don't write
python -m scripts.experiment.curation.refresh_service_support --no-backup # skip the backup file
```

The diff printed after each run lists which language codes were added or removed since the previous snapshot, per service.

The HTML marketing/documentation page at `cloud.google.com/translate/docs/languages` is intentionally not used as a fallback — its CSS-class layout is fragile to scrape; the v3 API returns a stable, structured list and survives Google's UI redesigns. The trade-off: the v3 API's supported-languages response can include languages the older v2 API rejects (see the `zh-min-nan` finding in `spotcheck_bcp47_codes.py`), so the `*_supported` flags downstream are accurate for v3 but mildly optimistic relative to v2.

---

### `spotcheck_bcp47_codes.py`

Supplementary validation of the BCP 47 enrichment layer. For each row where `bcp47_tag` substantively differs from `language_code` (excluding pure hyphen/case normalization), this script queries Google Translate's v2 API under both forms for a test term and reports whether the service returns identical, divergent, or asymmetric (one-form-fails) translations.

This is an *additive* experiment, not a rerun: the pipeline's translations remain keyed off `language_code` (the community/registry identifier). The script answers the methodological question "does normalization change anything operationally?"

```bash
python -m scripts.experiment.curation.spotcheck_bcp47_codes
python -m scripts.experiment.curation.spotcheck_bcp47_codes --term "Digital Humanities"
python -m scripts.experiment.curation.spotcheck_bcp47_codes \
    --csv-out datasets/metadata_files/bcp47_spotcheck_results.csv
```

Outcome classes per row:

| Outcome | Meaning |
| --- | --- |
| `agree` | Both forms accepted; identical translation. Normalization is semantically a no-op. |
| `diverge` | Both forms accepted; **different** translations. Paper-grade finding. |
| `project_only` | `language_code` accepted, `bcp47_tag` rejected. Community identifier is what's actually deployed. |
| `bcp47_only` | `bcp47_tag` accepted, `language_code` rejected. Google has standardized to the IANA form. |
| `both_failed` | Neither form supported. Expected for most grandfathered Wikimedia tags. |

Committed results live at `datasets/metadata_files/bcp47_spotcheck_results.csv`; notebook 01 §1.1 reads from that file. The 2026-06 run produced 5 `agree`, 1 `bcp47_only` (`map-bms` → `jv`, returning a Javanese translation — a fallback-overreach finding), and 7 `both_failed` (the grandfathered Wikimedia tags plus `cnr` Montenegrin, which lacks Google MT support under either identifier).

---

### `generate_family_assignments.py`

One-time generator that produced `datasets/metadata_files/language_family_assignments.json` — the JSON fallback used by `add_family_info()` in the build script for the long tail of codes not covered by `MANUAL_LANG_TO_SET5` or by CLDR's `languageGroups.json` direct walk-up. Each entry records a `language_code`, `family_name`, and `iso639_5` code where applicable, along with a brief rationale.

This script is **not part of the active pipeline** — it ran once and the JSON is committed. It is kept as the authoritative source for the family assignment decisions: if you ever need to add or revise assignments (e.g. after expanding the language list with new codes that none of the upstream registries cover), edit the `ASSIGNMENTS` dict in this script and re-run to regenerate the JSON. Then re-run `generate_language_codes.py` so the comprehensive CSV picks up the new assignments.

```bash
python -m scripts.experiment.curation.generate_family_assignments
```
