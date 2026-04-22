# Exploration

Scripts for exploring translation disagreements and measuring consistency across services and prompt variants. The goal is not to determine a "correct" translation but to understand *why* disagreements occur.

## Workflow Overview

```text
experiment/    → raw translations, fully automated, no human loop
                         ↓
exploration/
  Step 1       explore_confidence_within_variant.py
               → confidence_scores.csv (LLM agreement per variant)
                         ↓
  Step 2       explore_confidence_across_variants.py
               → across_variant_detail.csv (prompt robustness per service)
                         ↓
  Step 3       explore_disagreements.py
               → disagreement_analysis.csv (typology classification)
```

---

## Files

### `explore_confidence_within_variant.py` — Step 1: Within-variant LLM agreement

For each prompt variant separately, measures how much the four LLM services (OpenAI, Claude, Gemini, Ollama) agree on the same translation.

- **Question**: "Within a single variant, do the LLMs converge on the same translation?"
- **Input**: `translated_terms/{term}/direct_services/` and `translated_terms/{term}/prompt_services/`
- **Output**: `translated_terms/{term}/evaluation/confidence_scores.csv`, `confidence_summary.csv`

Confidence = fraction of services agreeing on the most common translation. Baseline services (GT, EasyNMT, Lingvanex, Wikipedia) are tracked separately as a prompt-invariant reference.

```bash
python explore_confidence_within_variant.py --term "Digital Humanities"
python explore_confidence_within_variant.py --variants minimal expert_persona
```

---

### `explore_confidence_across_variants.py` — Step 2: Cross-variant prompt robustness

For each (language × service), compares translations across all five prompt variants. Measures whether a given LLM gives the same answer regardless of how it was asked.

- **Question**: "Does the same LLM produce the same translation for 'expert_persona' and 'minimal'?"
- **Input**: `translated_terms/{term}/prompt_services/` for each variant
- **Output**: `translated_terms/{term}/evaluation/across_variant_detail.csv`, `across_variant_service_summary.csv`

Agreement rate = fraction of variants that produced the same translation for a given (language × service). Low agreement = the model has no confident grounded answer and framing shifts the output.

```bash
python explore_confidence_across_variants.py --term "Digital Humanities"
python explore_confidence_across_variants.py --variants minimal comparative expert_persona
```

---

### `explore_disagreements.py` — Step 3: Disagreement typology

Classifies each language into one of four disagreement categories using string matching on translations and rationale text:

| Category | Description |
| --- | --- |
| `NOT_APPLICABLE` | Only one service produced data — no disagreement possible |
| `MEASUREMENT_ARTEFACT` | Services agree after normalization (capitalization/whitespace only) |
| `PRODUCTIVE_DISAGREEMENT` | Wikipedia coverage exists; services debate between coexisting community terms |
| `STRUCTURAL_ABSENCE` | No established equivalent; services borrow the source term directly |
| `TRANSMOGRIFICATION` | No established equivalent; services construct elaborate novel terms |

- **Input**: `translated_terms/{term}/evaluation/across_variant_detail.csv` + variant DataFrames (for rationale columns)
- **Output**: `translated_terms/{term}/evaluation/disagreement_analysis.csv`

```bash
python explore_disagreements.py --term "Digital Humanities"
```

---

## Output Structure

All outputs are written to `translated_terms/{term}/evaluation/`:

```text
evaluation/
  confidence_scores.csv           # Step 1: per-row LLM agreement scores per variant
  confidence_summary.csv          # Step 1: aggregate stats per term/variant
  across_variant_detail.csv       # Step 2: per (language × service) agreement rate
  across_variant_service_summary.csv  # Step 2: aggregate stats per service
  disagreement_analysis.csv       # Step 3: typology classification per language
```

---

## Shared Dependencies

- `../../scripts/utils.py` — `get_data_directory_path`, `read_csv_file`, `LANGUAGE_FAMILIES`, `get_language_family`
