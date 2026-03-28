# Translation Generation Scripts

Scripts for generating translations using multiple services and LLM providers.

## Main Script
- **generate_translations.py** - Main orchestrator (moved to parent scripts/ for easy access)
  - Uses: data_processing.py, translation_services.py, verification.py

## Modules
- **data_processing.py** - Parsing, language detection, extraction utilities
- **translation_services.py** - API wrappers for all translation services
- **verification.py** - Term verification and validation functions
- **generate_translation_prompts.py** - Comparative testing of prompt variants

## Shared Dependencies
- `../translation_prompts.py` - Prompt templates
- `../utils.py` - Utility functions
