# Translation Pipeline Scripts

Scripts for generating and evaluating translations of Digital Humanities terms across 858 languages.

## Structure

```text
scripts/
  experiment/         - Active translation pipeline
  exploration/        - Analysis and disagreement exploration scripts
  utils.py            - Shared utilities (get_data_directory_path, read_csv_file, etc.)
```

## Language Coverage

Translations target all **858 languages** in `language_codes_comprehensive.csv`, built by combining four authoritative sources via `experiment/generate_language_codes.py`. The four non-language sentinel codes (`und`, `zxx`, `mis`, `mul`) are excluded; everything else is included for maximum coverage.

| Source | Coverage |
| --- | --- |
| Unicode CLDR | ~814 codes — scripts, directionality, official status |
| LOC ISO 639-2 | ~418 individual language codes |
| Wikimedia | ~270 active Wikipedia projects |
| ISO 639-5 | 115 family/group codes (metadata only, not translated) |

Most ancient, extinct, or low-resource languages will produce no results from direct MT services, but LLMs return translations for many of them. The `in_wikimedia`, `in_iso639_2`, and `modern_language` columns let downstream analysis filter to any subset.

## Prompt Variant Testing

LLM services are tested with four prompt strategies to evaluate the effect of different prompting approaches on translation quality:

| Variant | What it tests |
| --- | --- |
| `minimal` | Baseline — bare instruction, no context |
| `expert_persona` | Whether domain expert + native speaker framing improves accuracy |
| `native_rationale` | Whether requesting rationale in the target language deepens translation quality |
| `judge` | Synthesis — aggregates all unique translations from prior variants and direct services |

The first three variants are independent, directly comparable conditions. The `judge` runs last and is a synthesis step rather than a comparison condition: it receives all unique translations produced by the other variants and services, deduplicated by value with agreeing sources grouped, and produces a best-answer translation. See `experiment/README.md` for full details.

## Usage

**Build/refresh the language list:**

```bash
cd scripts/experiment
python generate_language_codes.py
```

**Generate translations (standard pipeline, minimal prompt):**

```bash
cd scripts/experiment
python generate_translations.py
```

**Run prompt variant testing:**

```bash
cd scripts/experiment
python generate_translation_prompts.py
```

Select individual variants or "ALL" from the interactive menu. When "ALL" is chosen, variants run in order: `minimal` → `expert_persona` → `native_rationale` → `judge`. The judge requires the other three to have completed first.
