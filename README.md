# Translation Transmogrification-Pipeline

A multilingual translation pipeline for scholarly terminology, built for the Coding DH project. Companion code for the DHQ article *"From Translation to Transmogrification: Exploring Multilingual Metadata and the Limits of AI-Augmented Translation in Digital Humanities"*.

## Overview

This pipeline translates key domain terms (e.g. "Digital Humanities", "Computational Social Science") into all ISO 639-1 languages by triangulating across multiple translation engines and LLMs, then surfaces disagreements for human verification.

**Translation engines:**
- Google Cloud Translate
- EasyNMT (Helsinki NLP opus-mt)
- Wikipedia language links (independent verification)
- OpenAI GPT-4o
- Anthropic Claude
- Ollama / Llama (local, run twice in comparative mode)

**Prompt variants tested:**
- `minimal` — bare-bones instruction
- `comparative` — shows prior engine outputs for refinement
- `expert_persona` — positions model as domain expert and native speaker
- `contextual` — provides rich semantic/definitional context
- `native_rationale` — asks for rationale in the target language

## Repository Structure

```
translation_pipeline/
├── models.py          # TranslationResponse, PipelineConfig, constants
├── parsers.py         # parse_translation_response, response extraction utilities
├── engines/
│   ├── google.py      # Google Cloud Translate
│   ├── enmt.py        # EasyNMT (opus-mt)
│   ├── openai.py      # OpenAI GPT-4o
│   ├── claude.py      # Anthropic Claude
│   ├── ollama.py      # Ollama / Llama (local)
│   └── wikipedia.py   # Wikipedia language links + directionality
├── pipeline/
│   ├── process.py     # Translation loop machinery (caching, error files, merging)
│   ├── combine.py     # Post-translation assembly, verification, output
│   └── generate.py    # Top-level orchestrators: generate_initial_terms, generate_translated_terms
├── verification/
│   └── html.py        # run_html_verification (browser-based review interface)
└── prompts/
    └── variants.py    # Five prompt variant functions + get_prompt dispatcher
```

## Quick Start

```python
from translation_pipeline.pipeline.generate import generate_translated_terms
from translation_pipeline.engines.wikipedia import get_directionality

directionality_df = get_directionality("path/to/directionality.csv")

term_contexts = {
    "Digital Humanities": (
        "'Digital Humanities' combines computational methods with the study of "
        "human culture, history, literature, and society."
    ),
}

translated_df, processed_df, grouped_df = generate_translated_terms(
    data_directory_path="path/to/data",
    target_terms=["Digital Humanities"],
    directionality_df=directionality_df,
    term_contexts=term_contexts,
    prompt_variant="comparative",
)
```

## Credentials

Requires API keys for Google Cloud Translate, OpenAI, and Anthropic, stored via the `apikey` library:

```bash
python -c "import apikey; apikey.save('GOOGLE_TRANSLATE_CREDENTIALS', 'path/to/credentials.json')"
python -c "import apikey; apikey.save('CODING_DH_OPENAI_KEY', 'your-key')"
python -c "import apikey; apikey.save('CODING_DH_CLAUDE_KEY', 'your-key')"
```

Ollama must be running locally: `ollama serve`

## Citation

If you use this pipeline, please cite:

> LeBlanc, Zoe. "From Translation to Transmogrification: Exploring Multilingual Metadata and the Limits of AI-Augmented Translation in Digital Humanities." *Digital Humanities Quarterly* (forthcoming).