# Translation Transmogrification-Pipeline

A multilingual translation pipeline for scholarly terminology, built for the Coding DH project. Companion code for the DHQ article *"From Translation to Transmogrification: Exploring Multilingual Metadata and the Limits of AI-Augmented Translation in Digital Humanities"*.

## Overview

This pipeline translates key domain terms (e.g. "Digital Humanities") into 880 language codes spanning ISO 639-1, ISO 639-2, and ISO 639-3, by triangulating across multiple translation services and LLMs with different prompting strategies, then surfaces and classifies disagreements for human verification.

**Direct services** (prompt-invariant, queried once per language):

- Google Cloud Translate
- EasyNMT (Helsinki-NLP opus-mt)
- Lingvanex
- Wikipedia title lookup

**LLM services** (8 models × 3 prompt variants = 24 outputs per language):

- Cloud: OpenAI GPT-4o, Anthropic Claude Sonnet 4.5, Google Gemini 2.5 Flash, DeepSeek V3
- Local (via Ollama): Llama 3.1, Gemma 3 (12B), Qwen 2.5 (7B), Mistral 7B

**Prompt variants:**

- `minimal` — bare instruction, no system prompt (baseline)
- `expert_persona` — positions model as domain expert and native speaker
- `native_rationale` — asks for rationale written in the target language
- `judge` — runs last; receives all unique translations from every prior service and variant, selects or improves the best one

## Documentation

- [docs/exclusion_strategy.md](docs/exclusion_strategy.md) — Data quality flags, manual exclusion taxonomy, and the three-tier exclusion policy (service exploration → translation analysis → search term generation)
- [html_files/five_source_language_code_pipeline.svg](html_files/five_source_language_code_pipeline.svg) — How the 880-language target set is constructed from five registries
- [html_files/translation_pipeline_methods.svg](html_files/translation_pipeline_methods.svg) — Pipeline architecture diagram

## Repository Structure

```text
translation_transmogrification_pipeline/
├── notebooks/                          # Analysis notebooks (run in order)
│   ├── 01_language_exploration.ipynb   # Language set construction and source coverage
│   ├── 02_translation_overview.ipynb   # Quality flags, exclusions, correction patterns
│   ├── 03_baseline_services.ipynb      # Direct service analysis
│   ├── 04_prompt_services.ipynb        # LLM service and prompt variant analysis
│   ├── 05_disagreement_analysis.ipynb  # Cross-service disagreement classification
│   ├── 06_translation_projection.ipynb # Embedding projections and family clustering
│   ├── 07_rationale_classification.ipynb # LLM rationale quality analysis
│   └── 08_search_results.ipynb         # GitHub search coverage and language distribution
├── scripts/
│   ├── utils.py                        # Shared utilities (load_manual_exclusions, etc.)
│   ├── generate_search_terms.py        # Build grouped_translated_terms.csv for search
│   ├── experiment/                     # Translation generation scripts
│   │   ├── generate_language_codes.py  # Build language_codes_comprehensive.csv
│   │   ├── generate_translations.py    # Main translation loop
│   │   ├── translation_services.py     # Per-service API calls
│   │   └── translation_prompts.py      # Prompt variant definitions
│   └── exploration/                    # Post-translation analysis scripts
│       ├── explore_disagreements.py    # Disagreement classification
│       ├── build_review_explorer_data.py
│       └── build_disagreement_explorer_data.py
├── datasets/
│   ├── metadata_files/                 # Language codes, family assignments, scripts
│   └── translated_terms/{term_slug}/
│       ├── direct_services/            # GT, EasyNMT, Lingvanex, Wikipedia outputs
│       ├── prompt_services/            # LLM × variant outputs
│       ├── evaluation/                 # Quality flags, disagreement analysis, exclusions
│       └── search_terms/               # reviewed_grouped_translated_terms.csv
├── html_files/                         # Interactive HTML review tools and figures
│   ├── review_explorer_v2.html         # Manual exclusion and term correction interface
│   └── search_term_reviewer.html       # Search term keep/exclude review interface
└── docs/
    └── exclusion_strategy.md
```

## Workflow

1. **Build language set** — `scripts/experiment/generate_language_codes.py` merges CLDR 48.2, CLDR v45 supplement, LOC ISO 639-2, and Wikimedia into `language_codes_comprehensive.csv` (880 codes).
2. **Run translations** — `scripts/experiment/generate_translations.py` queries all services and variants per term × language pair.
3. **Analyse** — run notebooks 01–07 in order to generate quality flags, disagreement analysis, embeddings, and rationale classifications.
4. **Review** — open `html_files/review_explorer_v2.html` to apply manual exclusions and term corrections; results saved to `evaluation/manual_exclusions.csv`.
5. **Generate search terms** — `scripts/generate_search_terms.py` applies all exclusions and produces `grouped_translated_terms.csv`; review in `html_files/search_term_reviewer.html`.
6. **Search** — feed `reviewed_grouped_translated_terms.csv` to the companion [searching_for_DH](https://github.com/CodingDH/searching_for_DH) pipeline.
7. **Search results** — run notebook 08 to analyse GitHub coverage and language distribution.

## Credentials

API keys are stored via the `apikey` library:

```bash
python -c "import apikey; apikey.save('GOOGLE_TRANSLATE_CREDENTIALS', 'path/to/credentials.json')"
python -c "import apikey; apikey.save('CODING_DH_OPENAI_KEY', 'your-key')"
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
python -c "import apikey; apikey.save('CODING_DH_DATA_DIRECTORY_PATH', 'path/to/datasets/')"
```

## Citation

If you use this pipeline, please cite:

> LeBlanc, Zoe. "From Translation to Transmogrification: Exploring Multilingual Metadata and the Limits of AI-Augmented Translation in Digital Humanities." *Digital Humanities Quarterly* (forthcoming).
