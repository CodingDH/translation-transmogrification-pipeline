# Translation Transmogrification Pipeline

A multilingual translation pipeline for scholarly terminology, built for the Coding DH project. Companion code for the DHQ article *"From Translation to Transmogrification: Exploring Multilingual Metadata and the Limits of AI-Augmented Translation in Digital Humanities"*.

## Overview

This pipeline translates key domain terms (e.g. "Digital Humanities") across 881 project language-code rows, built from ISO, CLDR, Wikimedia, SIL, Glottolog, and BCP 47/IANA-adjacent metadata layers, then surfaces and classifies disagreements for human verification.

**Machine translation baselines** (prompt-invariant, queried once per language):

- Google Cloud Translate
- EasyNMT (Helsinki-NLP opus-mt)
- Lingvanex

**Community-curated references** (human editorial consensus, not machine translation):

- Wikipedia interlanguage links — what scholars/editors in each language community have decided this concept is called
- *(Extensible — future additions could include IATE, LCSH, DH journal subject headings, Wikidata, etc.)*

**LLM services** (8 models × 4 prompt variants = up to 32 outputs per language):

- Cloud: OpenAI GPT-4o, Anthropic Claude Sonnet 4.5, Google Gemini 2.5 Flash, DeepSeek V3
- Local (via Ollama): Llama 3.1, Gemma 3 (12B), Qwen 2.5 (7B), Mistral 7B

**Prompt variants:**

- `minimal` — bare instruction, no system prompt (baseline)
- `fluent_speaker` — asks for rationale written in the target language
- `github_searcher` — frames translation as building a multilingual GitHub search corpus (goal-orientation)
- `judge` — runs last; receives all unique translations from every prior service and variant, selects or improves the best one

## Documentation

- [datasets/metadata_files/README.md](datasets/metadata_files/README.md) — Column-level reference for `language_codes_comprehensive.csv` and every other file in the metadata layer (provenance, BCP 47, service support, family reconciliation, dated snapshots)
- [scripts/experiment/README.md](scripts/experiment/README.md) — Per-script reference for the build, refresh, audit, and spot-check tools (`generate_language_codes`, `refresh_service_support`, `audit_language_identifiers`, `spotcheck_bcp47_codes`, etc.)
- [docs/exclusion_strategy.md](docs/exclusion_strategy.md) — Automated review signals, manual exclusion taxonomy, and the three-tier exclusion policy (service exploration → translation analysis → search term generation)
- [html_files/five_source_language_code_pipeline.svg](html_files/five_source_language_code_pipeline.svg) — How the 881-language target set is constructed from five registries
- [html_files/translation_pipeline_methods.svg](html_files/translation_pipeline_methods.svg) — Pipeline architecture diagram

## Repository Structure

```text
translation_transmogrification_pipeline/
├── notebooks/                          # Analysis notebooks (run in order)
│   ├── 01_language_exploration.ipynb   # Language set construction and source coverage
│   ├── 02_translation_overview.ipynb   # Automated review signals, exclusions, correction patterns
│   ├── 03_baseline_services.ipynb      # Direct service analysis
│   ├── 04_prompt_services.ipynb        # LLM service and prompt variant analysis
│   ├── 05_disagreement_analysis.ipynb  # Cross-service disagreement classification
│   ├── 06_translation_projection.ipynb # Embedding projections and family clustering
│   ├── 07_rationale_classification.ipynb # LLM rationale quality analysis
│   ├── 08_definitional_patterns.ipynb  # What LLMs encode about "Digital Humanities" (rationale-based)
│   └── 09_search_results.ipynb         # GitHub search coverage and language distribution
├── scripts/
│   ├── utils.py                        # Shared utilities (load_manual_exclusions, etc.)
│   ├── generate_search_terms.py        # Build grouped_translated_terms.csv for search
│   ├── experiment/                     # Translation generation scripts
│   │   ├── generate_language_codes.py  # Build language_codes_comprehensive.csv
│   │   ├── generate_translations.py    # Main translation loop
│   │   ├── translation_services.py     # Per-service API calls
│   │   └── translation_prompts.py      # Prompt variant definitions
│   └── exploration/                    # Post-translation analysis scripts
│       ├── build_family_*              # Family-label review and application
│       ├── build_review_explorer_data.py
│       ├── build_historic_reference_convergence.py
│       ├── explore_confidence_*.py     # Within/across prompt convergence
│       └── explore_disagreements.py    # Disagreement classification
├── datasets/
│   ├── metadata_files/                 # Language codes, family assignments, scripts
│   └── translated_terms/{term_slug}/
│       ├── direct_services/            # GT, EasyNMT, Lingvanex, Wikipedia outputs
│       ├── prompt_services/            # LLM × variant outputs
│       ├── evaluation/                 # Automated review signals, disagreement analysis, exclusions
│       └── search_terms/               # reviewed_grouped_translated_terms.csv
├── html_files/                         # Interactive HTML review tools and figures
│   ├── review_explorer.html            # Manual exclusion and term correction interface
│   └── search_term_reviewer.html       # Search term keep/exclude review interface
└── docs/
    └── exclusion_strategy.md
```

## Workflow

1. **Build language set** — `scripts/experiment/generate_language_codes.py` merges CLDR 48.2, CLDR v45 supplement, LOC ISO 639-2, and Wikimedia into `language_codes_comprehensive.csv` (881 codes). The same script also adds a parallel interoperability layer via `add_identifier_context()` and `add_service_language_codes()`:
   - **BCP 47 identifier columns** (`bcp47_tag`, `bcp47_source`, `bcp47_note`, `wikimedia_code`, `service_code_candidates`) provide preferred IANA-registry web tags alongside the project's stable `language_code`. Canonicalization is delegated to the [`langcodes`](https://github.com/georgkrause/langcodes) library (which validates against the IANA Language Subtag Registry shipped via `language_data`); a small `BCP47_GRANDFATHERED_FALLBACKS` dict in `generate_language_codes.py` handles seven Wikimedia legacy tags absent from the 2021-08-06 IANA snapshot. The columns do not change row identity and intentionally do not expand the dataset into BCP 47 locale variants (en-US, en-GB, es-MX, etc.) — the unit of analysis for this project is language/community discovery, not locale-specific localization. A 13-row supplementary spot-check (run via `scripts/experiment/curation/spotcheck_bcp47_codes.py`, results committed at `datasets/metadata_files/bcp47_spotcheck_results.csv`) confirmed that for every row where Google Translate accepts both the `language_code` and the canonical `bcp47_tag` form, the returned translation is identical — so the BCP 47 layer functions as interoperability documentation rather than as an alternative query path, and the pipeline's translations are not affected by the choice between forms. See nb01 §1.1 for the per-row results and the few asymmetric cases (`map-bms` → `jv` overreach; `cnr` Montenegrin unsupported under either form).
   - **Service-support columns** (`google_nmt_code`, `google_nmt_supported`, `google_translation_llm_code`, `google_translation_llm_supported`) map each language to the exact code the named service accepts, drawing from `datasets/metadata_files/service_language_code_support.csv`. Of the 881 pipeline languages, ~184 are officially supported by Google NMT and ~87 by Google's Translation LLM endpoint; the gap is itself a finding documenting the long tail of unsupported scholarly language communities. Use `scripts/experiment/curation/audit_language_identifiers.py` to inspect the layer at any time. The support CSV is a dated snapshot — refresh it with `python -m scripts.experiment.curation.refresh_service_support`, which calls the Translation v3 `get_supported_languages` API and stamps `snapshot_date` / `snapshot_source` on every row (NMT on the `global` endpoint, Translation LLM on `us-central1`). Previous snapshots are backed up alongside the file so paper-time analyses stay reproducible against their original snapshot date.
2. **Curate family classifications** — run notebook 01 §1.3 to cross-validate the ISO 639-5 based `family_name` column against Glottolog 5.3 ([CC-BY-4.0](https://glottolog.org/), cached at `datasets/metadata_files/glottolog-cache/`). The generated language list remains `language_codes_comprehensive.csv`; family review is applied as a later analytical metadata layer. First run `scripts/exploration/build_family_analysis_mapping.py` and review `family_analysis_mapping.csv` in `html_files/family_analysis_mapping_reviewer.html` to set the family-label policy. Then run `scripts/exploration/build_family_reconciliation_review.py`; it creates `family_reconciliation_review.csv`, a unified review queue combining ISO/Glottolog disagreement pairs with JSON fallback language rows and analysis-policy flags. Review it in `html_files/family_reconciliation_reviewer.html`, save the download as `datasets/metadata_files/family_reconciliation_reviewed.csv`, then run `scripts/exploration/apply_family_review_metadata.py`. This writes `language_codes_comprehensive_family_reviewed.csv`, preserving the raw `family_name` trace while adding `family_name_reconciled_reviewed` and `family_name_analysis`. The `get_language_family()` utility prefers this reviewed file when present.
3. **Run translations** — `scripts/experiment/generate_translations.py` queries all services and variants per term × language pair.
4. **Analyse** — run notebooks 02–08 in order to generate automated review signals, disagreement analysis, embeddings, rationale classifications, and definitional-pattern analysis. Before running notebook 02's historic reference-convergence section, refresh its precomputed tables with `python scripts/exploration/build_historic_reference_convergence.py --term "Digital Humanities"`; this reads `historic_materials/translated_dh_terms.csv` and writes `historic_*.csv` outputs to `datasets/translated_terms/digital_humanities/evaluation/`.
5. **Review** — open `html_files/review_explorer.html` to apply manual exclusions and term corrections; results saved to `evaluation/manual_exclusions.csv`.
6. **Generate search terms** — `scripts/generate_search_terms.py` applies all exclusions and produces `grouped_translated_terms.csv`; review in `html_files/search_term_reviewer.html`.
7. **Search** — feed `reviewed_grouped_translated_terms.csv` to the companion [searching_for_DH](https://github.com/CodingDH/searching_for_DH) pipeline.
8. **Search results** — run notebook 09 to analyse GitHub coverage and language distribution.

## Credentials

API keys are stored via the `apikey` library:

```bash
python -c "import apikey; apikey.save('GOOGLE_TRANSLATE_CREDENTIALS', 'path/to/credentials.json')"
python -c "import apikey; apikey.save('CODING_DH_OPENAI_KEY', 'your-key')"
python -c "import apikey; apikey.save('CODING_DH_OPENAI_PROJECT_ID', 'your-project-id')"
python -c "import apikey; apikey.save('CODING_DH_OPENAI_ORGANIZATION_ID', 'your-org-id')"
python -c "import apikey; apikey.save('CODING_DH_CLAUDE_KEY', 'your-key')"
python -c "import apikey; apikey.save('CODING_DH_GEMINI_KEY', 'your-key')"
python -c "import apikey; apikey.save('CODING_DH_DEEPSEEK_KEY', 'your-key')"
```

Local models require Ollama running with the relevant models pulled:

```bash
ollama serve
ollama pull llama3.1
ollama pull gemma3:12b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

Data directory paths are also stored via `apikey`:

```bash
python -c "import apikey; apikey.save('TRANSLATION_TRANSMOGRIFICATION_DATA_DIRECTORY', 'path/to/datasets/')"
```

## Citation

If you use this pipeline, please cite:

> LeBlanc, Zoe. "From Translation to Transmogrification: Exploring Multilingual Metadata and the Limits of AI-Augmented Translation in Digital Humanities." *Digital Humanities Quarterly* (forthcoming).
