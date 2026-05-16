# experiment/

Active translation pipeline — generation, prompt variant testing, and output verification.

## Files

### `generate_language_codes.py`

Builds `language_codes_comprehensive.csv` by combining five sources:

1. **Unicode CLDR 48.2** (via npm `cldr-core` + `cldr-localenames-full`) — scripts, directionality, modern-use status, official-language status, and family hierarchy (~802 language codes). Downloaded from the npm registry and cached locally; no HTML scraping.
2. **LOC ISO 639-2** — ISO 639-1/2 code mappings for individual languages (~190 codes).
3. **Wikimedia** — languages with active Wikipedia projects (~270 codes); proxy for community digital presence.
4. **CLDR v45 supplement** — 23 ISO 639-3 languages present in CLDR 45 but removed from CLDR 48.2 because their communities stopped submitting locale data required for software internationalisation. That criterion does not apply to scholarly translation. All 23 are valid living languages; they are marked `sources='cldr_v45'` and defined in the `CLDR_V45_SUPPLEMENT` constant. See the [CLDR 47 release notes](https://cldr.unicode.org/downloads/cldr-47) for the locale-removal policy.
5. **SIL ISO 639-3** — English reference names for the ~190 CLDR codes that have no entry in CLDR's `en/languages.json` (CLDR's English locale-names file covers ~693 of its ~802 codes; the rest fell through to code-as-name). The SIL table covers 214 of the 216 missing codes; the remaining two (`kro` = "Kru languages", `tokipona` = "Toki Pona") are filled from the hardcoded `_SIL_NAME_OVERRIDES` constant. The SIL file is downloaded once and cached alongside the CLDR npm packages.

Family hierarchy (Step 4) also comes from CLDR's `languageGroups.json`, which is part of the same `cldr-core` package fetched in Step 1 — no additional download or Wikipedia scraping.

`load_language_codes()` is the single entry point for the rest of the pipeline. It returns all languages in the merged CSV (~880 codes), excluding only the four non-language sentinel codes (`und`, `zxx`, `mis`, `mul`). This gives the broadest defensible translation target set: most ancient or low-resource languages will return no results from direct MT services, but LLMs produce translations for many of them.

All 880 codes in the output CSV have a non-empty value for every key column: `language_code`, `language_name` (English), `family_name`, `directionality`, and `sources`.

Directionality is sourced from CLDR's primary script code (authoritative), with `FORCE_LTR` overrides for languages whose historical script was RTL but whose modern standard is Latin or Cyrillic (e.g. Uzbek, Turkish). A separate `directionality_wikimedia` column preserves the Wikimedia community value for spotting digraphia cases.

Run standalone to regenerate the language list:

```bash
python generate_language_codes.py
# optionally pin a different CLDR version or cache location:
python generate_language_codes.py --cldr-version 48.2.0 --cldr-cache-dir /path/to/cache
```

Output is written to `{DATA_DIR}/metadata_files/`:

- `language_codes_comprehensive.csv` — one row per language code (with `cldr_version` provenance column)
- `language_scripts_long.csv` — one row per language × script pair
- `iso_639_set5.csv` — CLDR language group codes with parent and depth info
- `cldr-cache/` — cached npm packages and SIL ISO 639-3 table (not committed; re-used on subsequent runs)

---

### `parse_cldr_json.py`

CLDR data fetcher used by `generate_language_codes.py`. Downloads and parses the `cldr-core` and `cldr-localenames-full` npm packages, then exposes three functions:

- `parse_cldr_json()` — language codes with scripts, directionality, modern-use, official status
- `parse_cldr_family_groups()` — language family hierarchy compatible with `add_family_info()`
- `parse_cldr_aliases()` — deprecated/legacy code mappings (e.g. `iw` → `he`)

Both npm packages are cached after the first download; subsequent runs are fully offline. See the module docstring for citation guidance.

> **Removed:** `parse_iso639_5()` — previously scraped the ISO 639-5 family hierarchy from Wikipedia. Removed when the pipeline switched to CLDR 48.2 JSON; family hierarchy now comes from `parse_cldr_family_groups()` in `parse_cldr_json.py`, which reads `languageGroups.json` from the already-cached `cldr-core` package. The output file is still named `iso_639_set5.csv` for compatibility with `load_language_codes()`.

---

### `generate_family_assignments.py`

One-time generator that produced `datasets/metadata_files/language_family_assignments.json` — the JSON fallback used by `load_language_codes()` for the ~610 language codes not covered by `MANUAL_LANG_TO_SET5`. Each entry records a `language_code`, `family_name`, and `iso639_5` code where applicable, along with a brief rationale.

This script is **not part of the active pipeline** — it ran once and the JSON is committed. It is kept as the authoritative source for the family assignment decisions: if you ever need to add or revise assignments (e.g. after expanding the language list), edit the `ASSIGNMENTS` dict here and re-run to regenerate the JSON.

```bash
python generate_family_assignments.py
```

---

### `generate_translations.py`

Main pipeline orchestrator. Loops over target terms and runs all enabled translation services, returning three DataFrames: raw translations, processed/verified translations, and grouped-by-term summaries.

Key functions:

- `generate_translated_terms()` — top-level entry point; accepts boolean flags to enable/disable each service
- `generate_initial_terms()` — runs each service in order and merges results
- `combine_language_data()` — merges language metadata with translations and runs verification
- `save_translated_terms()` — writes per-service CSVs with path routing based on `prompt_variant`

Services run in this order (each optional):

| # | Service | Type | Prompt-variant? |
| --- | --- | --- | --- |
| 1 | Wikipedia | Ground truth lookup | No |
| 2 | Google Cloud Translate | Direct MT | No |
| 3 | EasyNMT | Local neural MT | No |
| 4 | Lingvanex | Direct MT (Kraus et al. 2025 top-ranked) | No |
| 5 | Llama (local Ollama) | Local LLM | Yes |
| 6 | Gemma (local Ollama) | Local LLM | Yes |
| 7 | Qwen (local Ollama) | Local LLM | Yes |
| 8 | Mistral (local Ollama) | Local LLM | Yes |
| 9 | OpenAI (GPT-4o) | Cloud LLM | Yes |
| 10 | Claude | Cloud LLM | Yes |
| 11 | Gemini (Kraus et al. 2025 top-ranked) | Cloud LLM | Yes |
| 12 | DeepSeek (V3) | Cloud LLM | Yes |

Translation priority for the final `term` column: Wikipedia → Gemini → OpenAI → Claude → DeepSeek → Llama → Gemma → Qwen → Mistral → Lingvanex → Google Translate → EasyNMT.

The default `prompt_variant` when running `generate_translations.py` standalone is `minimal`.

**CLI flags** — the `__main__` block accepts:

```bash
# Run multiple variants in sequence
python generate_translations.py --terms "Digital Humanities" --variant minimal expert_persona native_rationale judge

# Run only local Ollama models (skips all API LLMs and baseline services)
python generate_translations.py --terms "Digital Humanities" --ollama-only --variant minimal expert_persona native_rationale judge

# Run only cloud API models (skips all Ollama models and baseline services)
python generate_translations.py --terms "Digital Humanities" --api-only --variant minimal expert_persona native_rationale judge

# Fine-grained per-service control
python generate_translations.py --terms "Digital Humanities" --variant expert_persona --no-gt --no-enmt --no-lingvanex --no-wikipedia --no-openai --no-claude --no-gemini --no-deepseek
```

`--ollama-only` and `--api-only` are mutually exclusive. Both implicitly skip the baseline services (Wikipedia, GT, EasyNMT, Lingvanex) since those are prompt-invariant and only need to run once.

---

### `generate_translation_prompts.py`

Runner for prompt variant testing. Runs four strategies across all LLM services (OpenAI, Claude, Gemini, Ollama), saving results to separate per-variant files. Not part of the standard pipeline run — used for evaluating prompt engineering impact.

**The `judge` variant always runs last** and cannot be run meaningfully in isolation: it depends on the other three variants having already produced output files on disk. The script warns and aborts the judge run if no prior outputs are found.

Run order when "ALL" is selected:

```text
1. minimal  →  2. expert_persona  →  3. native_rationale  →  [aggregate]  →  4. judge
```

The aggregation step (`aggregate_variant_translations()`) loads every output file from variants 1–3 plus all direct service files, deduplicates translations by value, groups agreeing sources, and builds a per-(language, term) context dict that is passed to the judge LLMs.

---

### `translation_prompts.py`

Prompt templates for LLM services. Four variants, always run in this order:

| Variant | Strategy | What it tests |
| --- | --- | --- |
| `minimal` | Bare instruction with no additional context | Baseline — how well LLMs translate without guidance |
| `expert_persona` | Model positioned as domain expert and native speaker | Whether role/identity framing improves accuracy |
| `native_rationale` | Rationale requested in the target language | Whether metacognitive depth in the target language improves translation |
| `judge` | Synthesis of all prior outputs (runs last) | Best achievable translation given maximum available evidence |

**Design rationale:** The first three variants are independent conditions testing distinct prompting dimensions. They are directly comparable to each other. The `judge` is a synthesis step, not a comparison condition — its outputs should not be compared against the other three statistically. It answers the question "what is the best translation we can produce?" by aggregating all available evidence.

**Judge context format:** For each (language, term) pair, the judge receives:

- All unique translation values produced across variants 1–3 and all direct services, grouped by identical value so consensus is visible (e.g. "Translation 1 (Minimal — OpenAI, Expert persona — Claude): ...")
- A brief description of each source approach so the judge understands what produced each candidate
- A note listing any sources that produced no translation for this language — useful signal for low-resource languages

All variants share the same JSON response format: `{"translated_term": "...", "translation_rationale": "..."}`.

---

### `data_processing.py`

Parsing and extraction utilities.

- `TranslationResponse` — Pydantic model for structured LLM output
- `parse_translation_response()` — robustly parses JSON from LLM responses (handles markdown fences, single quotes, near-miss key names, Gemini newline escaping)
- `extract_dictionaries_from_string()` — extracts dicts from raw Ollama output
- `extract_ollama_translated_term()` — picks the best translation from Ollama's extracted dicts
- `is_enmt_model_available()` — checks if EasyNMT has a Helsinki-NLP model for a target language

---

### `translation_services.py`

API wrappers for every translation service. Each function takes `(row, error_file_path, console, ...)` and returns the row with translation and prompt columns filled in. System and user prompts are saved alongside translations for audit.

- `check_if_wikipedia_page_exists()` — fetches Wikipedia page translations across all languages
- `get_gt_translation()` — Google Cloud Translate
- `get_enmt_translation()` — EasyNMT (local)
- `get_lingvanex_translation()` — Lingvanex API
- `get_openai_translation()` — OpenAI GPT-4o
- `get_claude_translation()` — Anthropic Claude
- `get_gemini_translation()` — Google Gemini
- `get_deepseek_translation()` — DeepSeek V3
- `get_ollama_translation()` — local Ollama; shared by Llama, Gemma, Qwen, and Mistral via `_run_ollama_model()` in `generate_translations.py`

Context building for prompt variants is centralised in `_build_existing_translations()`: for `judge`, it looks up the pre-aggregated context from the `term_contexts` dict keyed by `(term_source, language_code)`. All other variants receive `None` (no external context).

---

### `verification.py`

Post-translation verification module. Currently a stub — term-level human review lives in `scripts/exploration/`.

---

## Error Logging

All service errors are written to `{DATA_DIR}/error_logs/` with one CSV per service (e.g. `openai_translation_errors.csv`). Each record includes `error_date`, `error_url`, `status_code`, `term_source`, `language_code`, and — for all LLM services — `variant`. `clean_write_error_file` deduplicates each log at run start using `(term_source, language_code, error_url, variant)` as the key; the `variant` field is ignored for baseline service logs that don't record it.

**Status codes determine whether a language is retried:**

| Status | Meaning | Behaviour |
| --- | --- | --- |
| 400 | Deterministic failure (unsupported language, model refusal, parse failure, hallucination loop) | Skip on all subsequent runs for this (term, language[, variant]) triple |
| 404 | Resource not found (model retired, Wikipedia page absent) | Skip permanently |
| 408 | Request timeout | Retry next run |
| 429 | Quota / rate limit exceeded | Aborts the entire service pass immediately |
| 500 | Unexpected exception (API error, network failure) | Retry next run |

**Parse failures are deterministic (400), not transient (500).** When an LLM returns prose instead of JSON — typically a refusal or a low-resource language the model does not recognise — `parse_translation_response` raises `ValueError`. This is caught explicitly in each service and logged as 400 so the language is not retried. A `ValueError` from a low-resource language will produce the same result on every retry; treating it as transient wastes API calls.

**Ollama hallucination loops are deterministic (400).** When a local model returns `done=False`, it hit the `num_predict` token limit mid-generation, producing a character-repetition loop rather than a valid translation. These are logged as `ollama.chat - HallucinationLoop` with status 400. The timing metadata fields (`total_duration`, etc.) are `None` in this state; `getattr` with a `None` default is used for all Ollama metadata to avoid `KeyError` from the library's `__getitem__` implementation, which omits `None`-valued fields.

**Errors are variant-scoped for LLM services.** A 400 failure on `minimal` does not exclude that language from `expert_persona`, `native_rationale`, or `judge`. Each prompt variant gets an independent exclusion check. This matters most for `judge`: low-resource languages that produced repetition loops or refusals on earlier variants still receive a judge attempt, since the judge prompt is structurally different and provides rich translation context from all prior outputs.

**Baseline service logs (GT, EasyNMT, Lingvanex, Wikipedia) have no variant column.** Their exclusions are unconditional — these services run once and are not re-run per variant. `log_error_to_file` always reindexes the new row to match the existing file's column order before appending, preventing column misalignment that would silently break the dedup key across runs.

The `exclude_{service}` boolean column is dropped before any output CSV is written.

## Output Structure

Per-term outputs under `{DATA_DIR}/translated_terms/{term_slug}/`:

```text
{term_slug}/
  direct_services/
    gt_translations.csv                      # Google Translate (prompt-invariant)
    enmt_translations.csv                    # EasyNMT (prompt-invariant)
    lingvanex_translations.csv               # Lingvanex (prompt-invariant)
    wikipedia_translations.csv               # Wikipedia (prompt-invariant)
  prompt_variants/
    llama_{variant}_translations.csv         # variant ∈ {minimal, expert_persona, native_rationale, judge}
    gemma_{variant}_translations.csv
    qwen_{variant}_translations.csv
    mistral_{variant}_translations.csv
    openai_{variant}_translations.csv
    claude_{variant}_translations.csv
    gemini_{variant}_translations.csv
    deepseek_{variant}_translations.csv
```
