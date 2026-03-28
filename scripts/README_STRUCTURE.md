# Scripts Organization Guide

## Directory Structure

```
scripts/
├── generate/                           # Translation generation pipeline
│   ├── __init__.py
│   ├── generate_translations.py        # Main orchestrator (refactored)
│   ├── data_processing.py              # Data parsing & utilities (param-based)
│   ├── translation_services.py         # API wrappers (param-based)
│   ├── verification.py                 # Term verification (param-based)
│   ├── generate_translation_prompts.py # Prompt variant testing
│   └── README.md
│
├── evaluate/                           # Translation evaluation & analysis
│   ├── __init__.py
│   ├── evaluate_confidence.py          # Frequency-based agreement scoring
│   ├── evaluate_github_alignment.py    # GitHub alignment analysis
│   ├── evaluate_meta_judge.py          # Meta-judge LLM evaluation
│   ├── evalute_llm_judge.py            # LLM translation judgment
│   └── README.md
│
├── generate_translations.py            # Symlink/shortcut to main script
├── translation_prompts.py              # Shared prompt templates
├── utils.py                            # Shared utilities
├── _old_versions/                      # Backup of pre-refactored modules
│   ├── data_processing.py
│   ├── translation_services.py
│   └── verification.py
│
└── README.md (this file)
```

## Key Changes Made

### ✅ Parameter-Based Architecture
- **Removed all `set_globals()` functions**
- Functions now accept parameters instead of relying on module-level globals
- Dependencies are explicit and testable

### ✅ Organized by Purpose
- **generate/** - Generation pipeline and utilities
- **evaluate/** - Standalone evaluation scripts
- **Shared modules** - Root-level utilities and templates

### ✅ Import Paths
From `scripts/generate/generate_translation_prompts.py`:
```python
# Add scripts/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now can import from scripts/
from generate_translations import generate_translated_terms
from translation_prompts import PROMPT_VARIANTS
from utils import get_data_directory_path
```

## Running Scripts

### Generate Translations
```bash
cd scripts
python generate_translations.py
# or
cd scripts/generate
python generate_translation_prompts.py
```

### Evaluate Translations
```bash
cd scripts/evaluate
python evaluate_confidence.py --term "Your Term"
python evaluate_github_alignment.py
python evaluate_meta_judge.py
python evalute_llm_judge.py  # (note: original typo preserved)
```

## Refactoring Summary

| Module | Status | Changes |
|--------|--------|---------|
| data_processing.py | ✅ Refactored | Params for `console`, `translate_client` |
| translation_services.py | ✅ Refactored | Params for all clients/models |
| verification.py | ✅ Refactored | Param for `console` |
| generate_translations.py | ✅ Updated | Updated imports & function calls |
| evaluate_*.py | ✅ Organized | Moved to `evaluate/`, no refactoring needed |
| generate_translation_prompts.py | ✅ Moved | Updated to `generate/` with fixed imports |
| translation_prompts.py | ✓ Shared | No changes needed |
| utils.py | ✓ Shared | No changes needed |

## Why This Structure?

1. **Clear Intent** - Scripts are grouped by what they do
2. **Dependency Management** - Parameters instead of globals = easier to test/maintain
3. **Scalability** - Easy to add new generation or evaluation methods
4. **Maintainability** - Changes to one service don't affect others
