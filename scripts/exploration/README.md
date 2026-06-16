# Exploration

Scripts for exploring translation disagreements and measuring consistency across services and prompt variants. The goal is not to determine a "correct" translation but to understand *why* disagreements occur.

> **Exclusion policy:** See [docs/exclusion_strategy.md](../../docs/exclusion_strategy.md) for the full taxonomy of automated review signals and the three-tier exclusion policy (service exploration → translation analysis → search term generation). The signals produced by `automated_review_signals.csv` and the manual review decisions in `manual_exclusions.csv` are applied differently at each tier.

## Workflow Overview

```text
experiment/    → raw translations, fully automated, no human loop
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
python explore_confidence_within_variant.py --term "Digital Humanities"
python explore_confidence_within_variant.py --variants minimal github_searcher
```

---

### `explore_confidence_across_variants.py` — Step 2: Cross-variant prompt robustness

For each (language × service), compares translations across the four prompt variants. Measures whether a given LLM gives the same answer regardless of how it was asked.

- **Question**: "Does the same LLM produce the same translation for 'github_searcher' and 'minimal'?"
- **Input**: `translated_terms/{term}/prompt_services/` for each variant
- **Output**: `translated_terms/{term}/evaluation/across_variant_detail.csv`, `across_variant_service_summary.csv`

Agreement rate = fraction of variants that produced the same translation for a given (language × service). Low agreement = the model has no confident grounded answer and framing shifts the output.

```bash
python explore_confidence_across_variants.py --term "Digital Humanities"
python explore_confidence_across_variants.py --variants minimal fluent_speaker github_searcher
```

---

### `explore_disagreements.py` — Step 3: Disagreement typology

Classifies each language into one of six disagreement categories using string matching on translations and rationale text:

| Category | Description |
| --- | --- |
| `TRANSMOGRIFICATION` | No established equivalent; services construct elaborate novel terms |
| `STRUCTURAL_ABSENCE` | No established equivalent; services borrow the source term directly |
| `PRODUCTIVE_DISAGREEMENT` | Wikipedia coverage exists; services debate between coexisting community terms |
| `MEASUREMENT_ARTEFACT` | Services agree after normalization (capitalization/whitespace only) |
| `NOT_APPLICABLE` | Only one service produced data — no disagreement possible |
| `CONSENSUS` | All services agree on the same translation without normalization |

- **Input**: `translated_terms/{term}/evaluation/across_variant_detail.csv` + variant DataFrames (for rationale columns)
- **Output**: `translated_terms/{term}/evaluation/disagreement_analysis.csv`

**Data quality — additional pairing enforcement.** Translations are sourced from `across_variant_detail.csv` (which aggregates across all variants), while rationales are loaded from a single specified variant file. A language can therefore appear in the detail CSV with a service translation even if that service has no rationale in the chosen variant. After loading both dicts, any LLM service that has a translation but no valid rationale in the loaded variant is dropped from the analysis — its output does not affect the classifier or rationale-similarity score. This is a second pass on top of the pairing already applied by `load_variant_df`.

**Manual exclusions.** Pass `--exclusions path/to/manual_exclusions.csv` (downloaded from `html_files/review_explorer.html` during the review stage) to suppress specific (language × service) translations or per-variant rationales from the analysis before classification runs.

```bash
python explore_disagreements.py --term "Digital Humanities"
python explore_disagreements.py --variant judge
python explore_disagreements.py --exclusions datasets/manual_exclusions.csv
```

---

## Output Structure

All outputs are written to `translated_terms/{term}/evaluation/`:

```text
evaluation/
  automated_review_signals.csv               # Review: per-language automated review signals (notebook 02); signals: missing_rationale, mixed_script, romanization, script_disagreement, source_term, placeholder_term, repetition_loop, extreme_term_length, unicode_escape
  review_explorer_data.csv        # Review: merged input for html_files/review_explorer.html
  manual_exclusions.csv           # Review: human decisions downloaded from review_explorer.html
  confidence_scores.csv           # Step 1: per-row LLM agreement scores per variant
  confidence_summary.csv          # Step 1: aggregate stats per term/variant
  across_variant_detail.csv       # Step 2: per (language × service) agreement rate
  across_variant_service_summary.csv  # Step 2: aggregate stats per service
  disagreement_analysis.csv       # Step 3: typology classification per language
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
