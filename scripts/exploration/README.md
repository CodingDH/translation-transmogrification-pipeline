# Exploration

Scripts for exploring translation disagreements and measuring consistency across services and prompt variants. The goal is not to determine a "correct" translation but to understand *why* disagreements occur.

> **Exclusion policy:** See [docs/exclusion_strategy.md](../../docs/exclusion_strategy.md) for the full taxonomy of automated review signals and the three-tier exclusion policy (service exploration → translation analysis → search term generation). The signals produced by `automated_review_signals.csv` and the manual review decisions in `manual_exclusions.csv` are applied differently at each tier.

## Workflow Overview

```text
experiment/    → raw translations, fully automated, no human loop
                         ↓
  Optional     build_historic_reference_convergence.py
  Historic     → historic_reference_*.csv
  Check        (used by notebook 02; historic Digital Humanities only)
                         ↓
notebooks/02_translation_overview.ipynb
               → automated_review_signals.csv   (per-language automated review signals)
                         ↓
  Review       build_review_explorer_data.py
               → review_explorer_data.csv
                         ↓
  Human        html_files/review_explorer.html
  Review       (load CSV, inspect flagged languages, mark exclusions)
               → manual_exclusions.csv   (downloaded from the HTML)
                         ↓
exploration/
  Step 1       explore_confidence_within_variant.py
               → confidence_scores.csv (LLM agreement per variant)
                         ↓
  Step 2       explore_confidence_across_variants.py
               → across_variant_detail.csv (prompt robustness per service)
                         ↓
  Step 3       explore_disagreements.py  [--exclusions manual_exclusions.csv]
               → disagreement_analysis.csv (typology classification)
```

The **review stage** sits between raw translation (notebook 02) and the analytical pipeline (Steps 1–3). It surfaces automated review signals — missing rationales, mixed-script output, placeholder/refusal terms, script disagreements between services, source-term leakage, repetition loops, and search-safety issues — before any downstream analysis runs. Human review decisions (service exclusions) are persisted as `manual_exclusions.csv` and passed to `explore_disagreements.py`.

The **historic check** is separate from the active translation-evaluation pipeline. It supports notebook 02's discussion of the original 185-language Digital Humanities translation artifact and should be run only when refreshing the historic-reference convergence tables.

---

## Files

### `translation_classifier.py` — Mixed-script and placeholder translation classifier

Classifies each translation string as `unchanged`, `stripped`, `nulled`, or `placeholder` based on script content and refusal detection. The four-action output propagates to `automated_review_signals.csv` (notebook 02), `review_explorer_data.csv`, and the HTML review explorer.

**Placeholder detection** runs first: if the text contains a refusal phrase (`"untranslatable"`, `"unable to translate"`, `"no direct translation"`, `"cannot be reliably translated"`, `"Note: ..."`, etc.) it is classified as `placeholder` and returned as `None` without entering the script-mixing logic. The regex also covers `"no confirmed/widely recognized/accepted translation"` and adverb-insertion variants like `"cannot be reliably translated"`.

**Script-mixing patterns** are then applied to remaining strings:

| Pattern | Description | Action |
| --- | --- | --- |
| A — Romanization helper | `"native_term (romanization)"` or `"native / romaji"` — model appended an unrequested Latin transliteration | `stripped` — parenthetical/slash removed, primary script kept |
| B — Character-level noise | Scripts interleaved mid-word with no structural delimiter | `nulled` — not salvageable |
| C — Colon-separated prefix | `"Digital Humanities : native_term"` — source term echoed before a colon | `stripped` — prefix discarded if it contains no dominant-script characters |
| D — Space-separated Latin prefix | `"Digital Humanities के दिशा कौशल"` — same as C but whitespace-delimited | `stripped` — leading Latin tokens stripped; non-Latin-leading strings are never touched |

Mixed-script detection uses a two-tier test designed to avoid false positives from orthographic extensions (Ossetian æ, Chuvash ă, Chechen palochka substitutes):

- **Tier 1** — ≥5 distinct minority-script characters → always flagged
- **Tier 2** — ≥3 distinct minority characters AND ≥25% of all script characters → flagged
- A single stray character representing <10% of the string is always ignored

**Script detection** delegates to `_char_script` in `scripts/utils.py`. When the `regex` package is available (the default), script identity is resolved via Unicode Script property patterns (`\p{Script=...}`) covering 55+ scripts including Thai, Lao, New Tai Lue, N'Ko, Vai, and other scripts that the old range-table missed. A `functools.lru_cache` prevents re-matching the same codepoint more than once per process. If `regex` is not installed the classifier falls back to an inline range table covering the 7 most common scripts.

**Importable API:** `curate_translation(text) → (result, action)`, `curate_df(df) → (curated_df, summary_df)`, `is_placeholder_term(text) → bool`, `is_repetition_loop(text) → bool`, `has_extreme_term_length(text, max_chars=100) → bool`, `has_unicode_escape(text) → bool`

---

### `build_review_explorer_data.py` — Review stage: build data for HTML explorer

Merges raw translations (prompt_services + direct_services) with `automated_review_signals.csv` (produced by notebook 02) into a single wide CSV for the review explorer. **No dependency on Steps 1–3** — this runs directly after notebook 02.

- **Input**: `translated_terms/{term}/evaluation/automated_review_signals.csv` + raw per-variant CSV files
- **Output**: `translated_terms/{term}/evaluation/review_explorer_data.csv`

Each row is one language. Columns include:

- Automated review signal booleans and which services triggered them: `has_missing_rationale`, `has_mixed_script`, `has_romanization`, `has_script_disagreement`, `has_source_term`, `has_placeholder_term`, `has_repetition_loop`, `has_extreme_term_length`, `has_unicode_escape`
- `review_tier`: `REVIEW_HIGH` (≥2 flags) / `REVIEW_MED` (1 flag) / `CLEAN` (0 flags)
- Per-service × per-variant term and rationale columns: `{svc}_term_{variant}`, `{svc}_rationale_{variant}`
- Non-LLM reference terms: `wikipedia_translated_term` (community reference), `gt_translated_term`, `enmt_translated_term`, `lingvanex_translated_term` (MT baselines)

Rows are sorted flagged-first so the HTML explorer opens on the cases that need attention.

```bash
python scripts/exploration/build_review_explorer_data.py --term "Digital Humanities"
python scripts/exploration/build_review_explorer_data.py --output-dir path/to/dir
```

---

### `build_historic_reference_convergence.py` — Notebook support: historic DH reference agreement

Builds the precomputed tables used by notebook 02's historic reference-convergence section. This script intentionally uses only `historic_materials/translated_dh_terms.csv` and hard-filters to `term_source == "Digital Humanities"` by default.

In that historic CSV, `term` is the pre-existing/Wikipedia-derived Digital Humanities term when one was available, while `translated_term` is the historic Google Translate output. The script therefore measures exact normalized agreement between those two historic fields; it does **not** use the legacy `old_digital_humanities` folder, and it does not claim to reconstruct OpenAI or EasyNMT convergence because those service-specific columns are not present in the historic file.

- **Input**: `historic_materials/translated_dh_terms.csv`
- **Output**: `translated_terms/digital_humanities/evaluation/historic_*.csv`
- **Default term filter**: `Digital Humanities`

```bash
python scripts/exploration/build_historic_reference_convergence.py --term "Digital Humanities"
```

Generated files:

| File | Purpose |
| --- | --- |
| `historic_reference_artifacts.csv` | Records the historic artifact used and its row/translation counts |
| `historic_reference_service_presence.csv` | Counts availability of the Wikipedia-derived term and historic GT output |
| `historic_pairwise_reference_agreement.csv` | Pairwise exact-normalized agreement between the two historic fields |
| `historic_reference_count_summary.csv` | Source availability and agreement by number of historic fields present |
| `historic_all_source_convergence_summary.csv` | Overall convergence categories: no outputs, one output only, agree, diverge |
| `historic_reference_convergence_detail.csv` | Per-language detail table used for examples in notebook 02 |

---

### `build_family_reconciliation_review.py` — Metadata review: unified family reconciliation queue

Builds `datasets/metadata_files/family_reconciliation_review.csv`, a unified review queue for family reconciliation. It combines ISO/Glottolog disagreement pairs from `family_reconciliation.csv` with JSON fallback language rows whose raw `family_name` assignment comes from `language_family_assignments.json` because no manual `MANUAL_LANG_TO_SET5` mapping wins.

Run this after generating and, if needed, reviewing `family_analysis_mapping.csv`; the builder annotates JSON fallback rows with that analysis-label policy so problematic labels can be corrected at the row level.

```bash
python scripts/exploration/build_family_reconciliation_review.py
```

Review tiers:

| Tier | Meaning |
| --- | --- |
| `REVIEW_HIGH` | The coherence layer flags a likely abandoned macrofamily, contested macrofamily, geographic grouping, special non-family category, or incorrect direct assignment |
| `REVIEW_MED` | No Glottolog match, Glottolog disagreed but the reconciliation decision kept the ISO/JSON family, CLDR has no ISO 639-5 hierarchy support, or a lower-confidence coherence issue needs spot-checking |
| `REFERENCE_ONLY` | Glottolog agreed or already reclassified the row; usually lower review priority |

The coherence columns (`coherence_issue`, `suggested_iso639_5`, `suggested_family_name`, `suggestion_confidence`, `suggestion_reason`) do not alter the pipeline output. They identify cases where the current JSON-derived family label may be too broad or unstable, such as abandoned macrofamilies (`Altaic`, `Niger-Kordofanian`), geographic groupings (`North American Indian languages`), or a known direct error (`frm` / Middle French).

The CSV includes editable browser-review columns: `reviewer_decision`, `reviewer_family_name`, `reviewer_iso639_5`, and `reviewer_notes`. The HTML reviewer downloads the edited file as `family_reconciliation_reviewed.csv`. When that reviewed file is saved in `datasets/metadata_files/`, run `apply_family_review_metadata.py` to apply reviewed pair decisions and JSON fallback rows where `reviewer_decision == "revise"` and `reviewer_family_name` is non-empty. This writes a reviewed metadata layer while leaving the generated language-code list unchanged.

---

### `build_family_analysis_mapping.py` — Metadata review: analysis family labels

Builds `datasets/metadata_files/family_analysis_mapping.csv`, one row per current `family_name_reconciled` value. The file proposes a `family_name_analysis` label for charts and aggregate comparison, preserving the reconciled Glottolog/review label by default. The HTML reviewer is for label-level decisions only: spelling variants, missing labels, and special non-family categories. Problematic macrofamily or geographic labels are marked `row_level_reconciliation` and passed to the unified reconciliation queue for language-level review.

```bash
python scripts/exploration/build_family_analysis_mapping.py
```

Review the generated CSV in `html_files/family_analysis_mapping_reviewer.html`. The reviewer groups rows by proposed analysis label, shows language counts, raw-family provenance, mapping actions, and sample languages, and downloads the edited file as `family_analysis_mapping_reviewed.csv`.

This file is not applied directly to `language_codes_comprehensive.csv`. It is the review surface for deciding analysis-label overrides. `apply_family_review_metadata.py` consumes the reviewed mapping later and writes the durable reviewed metadata layer, `language_codes_comprehensive_family_reviewed.csv`.

---

### `apply_family_review_metadata.py` — Metadata review: apply reviewed family layer

Applies reviewed family metadata after the language-code list has already been generated. This is the preferred way to use reviewed family names in downstream analysis without rerunning `generate_language_codes.py` or rewriting the source-build artifact.

```bash
python scripts/exploration/apply_family_review_metadata.py
```

Inputs:

| File | Purpose |
| --- | --- |
| `language_codes_comprehensive.csv` | Generated language-code/source-build artifact |
| `family_analysis_mapping_reviewed.csv` | Reviewed label policy for `family_name_analysis`; falls back to `family_analysis_mapping.csv` |
| `family_reconciliation_reviewed.csv` | Optional reviewed row/pair family corrections |

Output:

| File | Purpose |
| --- | --- |
| `language_codes_comprehensive_family_reviewed.csv` | Reviewed family metadata layer used by downstream `get_language_family()` calls when present |

The output preserves `family_name` and `family_name_reconciled_pre_review`, adds `family_name_reconciled_reviewed`, and writes `family_name_analysis` for chart/grouping work.

---

### `html_files/review_explorer.html` — Review stage: human review interface

Single-file HTML explorer for the review stage. Load `review_explorer_data.csv` via the "Load CSV" button.

**Navigation:** sidebar groups languages by review tier (REVIEW_HIGH → REVIEW_MED → CLEAN). Filter by tier, specific flag, or rationale text search.

**Card contents per language:**

- Automated review signal pills showing which flags fired and which services triggered them
- Term cluster panel — groups all 32 LLM cells (8 services × 4 variants) by normalized term; click to select
- Rationale matrix — full variants × services grid, click to expand rationale text
- Non-LLM reference terms — Wikipedia (community reference) and Google Translate / Lingvanex / EasyNMT (MT baselines)
- Service exclusions panel — toggle individual translations or per-variant rationales to exclude
- Notes field

**Outputs:**

- **Download Exclusions** → `manual_exclusions.csv` (pass to `explore_disagreements.py --exclusions`)
- **Export Selected** → `reviewed_terms.csv` (selected translation candidates with provenance)

State (selections, exclusions, notes) is persisted in `localStorage` between sessions.

---

### `explore_confidence_within_variant.py` — Step 1: Within-variant LLM agreement

For each prompt variant separately, measures how much the eight LLM services (OpenAI, Claude, Gemini, DeepSeek, Llama, Gemma, Qwen, Mistral) agree on the same translation. API services (OpenAI, Claude, Gemini, DeepSeek) and local Ollama services (Llama, Gemma, Qwen, Mistral) are both included.

- **Question**: "Within a single variant, do the LLMs converge on the same translation?"
- **Input**: `translated_terms/{term}/direct_services/` and `translated_terms/{term}/prompt_services/`
- **Output**: `translated_terms/{term}/evaluation/confidence_scores.csv`, `confidence_summary.csv`

Confidence = fraction of services agreeing on the most common translation. The four non-LLM sources — MT baselines (GT, EasyNMT, Lingvanex) and the Wikipedia community reference — are tracked separately as a prompt-invariant block.

**Data quality — translation/rationale pairing.** The central function exposed by this file, `load_variant_df`, is used by every downstream notebook and script to load per-variant data. Before returning, it calls `enforce_translation_rationale_pairing` (`scripts/utils.py`), which nulls out mismatched LLM columns:

| Failure mode | Example | Action |
| --- | --- | --- |
| Translation present, rationale absent or placeholder | Model returns term but omits explanation | Translation cell → `NaN` |
| Rationale present (or placeholder string), translation absent | Model explains but returns no term | Rationale cell → `NaN` |

Placeholder rationales — literal strings such as `"No rationale provided"`, `"N/A"`, `"None"` — are detected by `is_placeholder_rationale` (`scripts/utils.py`) and treated identically to `NaN`. This enforcement happens once at load time so all callers receive consistent, paired data.

```bash
python scripts/exploration/explore_confidence_within_variant.py --term "Digital Humanities"
python scripts/exploration/explore_confidence_within_variant.py --variants minimal github_searcher
```

---

### `explore_confidence_across_variants.py` — Step 2: Cross-variant prompt robustness

For each (language × service), compares translations across the four prompt variants. Measures whether a given LLM gives the same answer regardless of how it was asked.

- **Question**: "Does the same LLM produce the same translation for 'github_searcher' and 'minimal'?"
- **Input**: `translated_terms/{term}/prompt_services/` for each variant
- **Output**: `translated_terms/{term}/evaluation/across_variant_detail.csv`, `across_variant_service_summary.csv`

Agreement rate = fraction of variants that produced the same translation for a given (language × service). Low agreement means the service is prompt-sensitive for that language; it is a convergence signal, not direct proof that no grounded answer exists.

```bash
python scripts/exploration/explore_confidence_across_variants.py --term "Digital Humanities"
python scripts/exploration/explore_confidence_across_variants.py --variants minimal fluent_speaker github_searcher
```

---

### `explore_disagreements.py` — Step 3: Disagreement typology

Classifies each language into one of five rule-based buckets using string matching on translations, rationale text, and lightweight metadata:

| Category | Description |
| --- | --- |
| `COMPLETE_CONSENSUS` | All available services agree — no disagreement to classify |
| `MEASUREMENT_ARTEFACT` | Apparent disagreement dissolves after normalization or script-variant handling |
| `PRODUCTIVE_DISAGREEMENT` | Multiple in-language alternatives with a Wikipedia/community anchor |
| `STRUCTURAL_ABSENCE` | Source-term fallback or explicit absence signaling |
| `TRANSMOGRIFICATION` | No established equivalent; services construct elaborate novel terms |

- **Input**: `translated_terms/{term}/evaluation/across_variant_detail.csv` + variant DataFrames (for rationale columns)
- **Output**: `translated_terms/{term}/evaluation/disagreement_analysis.csv`

**Data quality — additional pairing enforcement.** Translations are sourced from `across_variant_detail.csv` (which aggregates across all variants), while rationales are loaded from a single specified variant file. A language can therefore appear in the detail CSV with a service translation even if that service has no rationale in the chosen variant. After loading both dicts, any LLM service that has a translation but no valid rationale in the loaded variant is dropped from the analysis — its output does not affect the classifier or rationale-similarity score. This is a second pass on top of the pairing already applied by `load_variant_df`.

**Manual exclusions.** Pass `--exclusions path/to/manual_exclusions.csv` (downloaded from `html_files/review_explorer.html` during the review stage) to suppress specific (language × service) translations or per-variant rationales from the analysis before classification runs.

```bash
python scripts/exploration/explore_disagreements.py --term "Digital Humanities"
python scripts/exploration/explore_disagreements.py --variant judge
python scripts/exploration/explore_disagreements.py --exclusions datasets/translated_terms/digital_humanities/evaluation/manual_exclusions.csv
```

---

### `build_disagreement_explorer_data.py` — Notebook support: disagreement analysis matrix

Builds `translated_terms/{term}/evaluation/disagreement_explorer_data.csv` after `explore_disagreements.py` has written `disagreement_analysis.csv`. Despite the historical script name, this file is no longer tied to a standalone HTML disagreement explorer. It is a notebook-support table consumed by notebook 07 for rationale classification and related inspection.

The script merges:

- `disagreement_analysis.csv` categories and rule evidence
- per-language confidence aggregates from `confidence_scores.csv`
- service translations parsed into separate columns
- term and rationale columns for each LLM service × prompt variant, with translation/rationale pairing enforced before export
- prompt-invariant baseline terms where available

```bash
python scripts/exploration/build_disagreement_explorer_data.py --term "Digital Humanities"
python scripts/exploration/build_disagreement_explorer_data.py --term "Digital Humanities" --rationale-variant fluent_speaker
```

---

### `rationale_classifier.py` — Notebook support: rationale genre classification

Importable helper functions for notebook 07. The classifier asks whether rationale prose reveals prompt variant or service identity after masking obvious leakage such as service names, prompt names, source-term tokens, language names, and translated terms.

This is not a standalone CLI step. Notebook 07 imports the masking and classification helpers to run the main rationale-genre analysis and masking ablations.

---

## Output Structure

All outputs are written to `translated_terms/{term}/evaluation/`:

```text
evaluation/
  historic_reference_artifacts.csv       # Historic check: input artifact summary for notebook 02
  historic_reference_service_presence.csv  # Historic check: availability of historic term/GT fields
  historic_pairwise_reference_agreement.csv  # Historic check: exact agreement between historic term and GT output
  historic_reference_count_summary.csv   # Historic check: availability/agreement by number of fields present
  historic_all_source_convergence_summary.csv  # Historic check: convergence categories
  historic_reference_convergence_detail.csv  # Historic check: per-language examples
  automated_review_signals.csv               # Review: per-language automated review signals (notebook 02); signals: missing_rationale, mixed_script, romanization, script_disagreement, source_term, placeholder_term, repetition_loop, extreme_term_length, unicode_escape
  review_explorer_data.csv        # Review: merged input for html_files/review_explorer.html
  manual_exclusions.csv           # Review: human decisions downloaded from review_explorer.html
  confidence_scores.csv           # Step 1: per-row LLM agreement scores per variant
  confidence_summary.csv          # Step 1: aggregate stats per term/variant
  across_variant_detail.csv       # Step 2: per (language × service) agreement rate
  across_variant_service_summary.csv  # Step 2: aggregate stats per service
  disagreement_analysis.csv       # Step 3: typology classification per language
  disagreement_analysis_no_keywords.csv  # Optional ablation output from explore_disagreements.py --no-keyword-rules
  disagreement_explorer_data.csv  # Notebook support: merged disagreement/confidence/rationale matrix for notebook 07
```

---

## Shared Dependencies

- `../../scripts/utils.py`

| Function | Used by | Purpose |
| --- | --- | --- |
| `get_data_directory_path` | all scripts | Locate the datasets/ root |
| `read_csv_file` | all scripts | Safe CSV loader with error handling |
| `get_language_family` | steps 1–3 | Map language code → family name |
| `_char_script(cp)` | `translation_classifier.py`, `detect_dominant_script` | Map a Unicode code point to a script family name. Uses `regex` `\p{Script=...}` patterns covering 55+ scripts when available; falls back to a range table for the 30 most common scripts. Results are `lru_cache`'d per process. |
| `detect_dominant_script(text)` | notebook 02 (§1.8.1 script disagreement) | Return the plurality script of a translation string, ignoring punctuation and digits |
| `is_placeholder_rationale` | step 1 (`load_variant_df`), step 3 | Detect literal placeholder strings returned instead of real rationales (e.g. `"No rationale provided"`, `"N/A"`) |
| `enforce_translation_rationale_pairing` | step 1 (`load_variant_df`), review (`build_review_explorer_data.py`) | Null out unpaired translation/rationale columns for all LLM services at load time |
