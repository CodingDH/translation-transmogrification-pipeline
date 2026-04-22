# experiment/

Active translation pipeline — generation, prompt variant testing, and output verification.

## Files

### `generate_language_codes.py`

Builds `language_codes_comprehensive.csv` by combining four authoritative sources:

1. **Unicode CLDR** — scripts, directionality, modern-use flag, official status (~814 codes)
2. **LOC ISO 639-2** — ISO 639-1/2 code mappings (~418 individual language codes)
3. **Wikimedia** — languages with active Wikipedia projects (~270 codes)
4. **ISO 639-5** — language family/group hierarchy (115 codes, metadata only — never translated)

`load_language_codes()` is the single entry point for the rest of the pipeline. It returns all **858 languages** in the merged CSV, excluding only the four non-language sentinel codes (`und`, `zxx`, `mis`, `mul`). This gives the broadest defensible translation target set: most ancient or low-resource languages will return no results from direct MT services, but LLMs produce translations for many of them.

Directionality is sourced from CLDR's primary script code (authoritative), with `FORCE_LTR` overrides for languages whose historical script was RTL but whose modern standard is Latin or Cyrillic (e.g. Uzbek, Turkish). A separate `directionality_wikimedia` column preserves the Wikimedia community value for spotting digraphia cases.

Run standalone to refresh the language list:

```bash
python generate_language_codes.py
```

Output is written to `{DATA_DIR}/metadata_files/`:

- `language_codes_comprehensive.csv` — one row per language code (858 total)
- `language_scripts_long.csv` — one row per language × script pair
- `iso_639_set5.csv` — 115 ISO 639-5 family/group codes with hierarchy

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
| 5 | Ollama | Local LLM baseline | Yes |
| 6 | OpenAI (GPT-4o) | Cloud LLM | Yes |
| 7 | Claude | Cloud LLM | Yes |
| 8 | Gemini (Kraus et al. 2025 top-ranked) | Cloud LLM | Yes |

Translation priority for the final `term` column: Wikipedia → Gemini → OpenAI → Claude → Ollama → Lingvanex → Google Translate → EasyNMT.

The default `prompt_variant` when running `generate_translations.py` standalone is `minimal`.

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
- `get_ollama_translation()` — local Ollama (llama3.1 by default)

Context building for prompt variants is centralised in `_build_existing_translations()`: for `judge`, it looks up the pre-aggregated context from the `term_contexts` dict keyed by `(term_source, language_code)`. All other variants receive `None` (no external context).

---

### `verification.py`

Post-translation verification module. Currently a stub — term-level human review lives in `scripts/exploration/`.

---

## Error Logging

All service errors are written to `{DATA_DIR}/error_logs/` with one CSV per service (e.g. `openai_translation_errors.csv`). Each record includes `error_date`, `error_url`, `status_code`, `term_source`, and `language_code`. `clean_write_error_file` deduplicates each log at run start.

**Transient errors (status 500, 408) are never used to permanently exclude a language** — they represent API failures, not unsupported languages. Only deterministic failures (400 language-not-supported, 404 model-not-found) result in that (language, term) pair being skipped on subsequent runs. The `exclude_{service}` boolean column is dropped before any output CSV is written.

## Output Structure

Per-term outputs under `{DATA_DIR}/translated_terms/{term_slug}/`:

```text
{term_slug}/
  direct_services/
    gt_translations.csv                      # Google Translate (prompt-invariant)
    enmt_translations.csv                    # EasyNMT (prompt-invariant)
    lingvanex_translations.csv               # Lingvanex (prompt-invariant)
    wikipedia_translations.csv               # Wikipedia (prompt-invariant)
  prompt_services/
    ollama_{variant}_translations.csv        # variant ∈ {minimal, expert_persona, native_rationale, judge}
    openai_{variant}_translations.csv
    claude_{variant}_translations.csv
    gemini_{variant}_translations.csv
```
