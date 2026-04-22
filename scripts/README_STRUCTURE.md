# Scripts Structure

```text
scripts/
├── experiment/                              # Active translation pipeline
│   ├── generate_language_codes.py           # Builds language_codes_comprehensive.csv
│   ├── generate_family_assignments.py       # Generated datasets/metadata_files/language_family_assignments.json (one-time)
│   ├── generate_translations.py             # Main orchestrator
│   ├── generate_translation_prompts.py      # Prompt variant testing runner
│   ├── translation_prompts.py               # Prompt templates (4 variants: minimal, expert_persona, native_rationale, judge)
│   ├── translation_services.py              # API wrappers for all services
│   ├── data_processing.py                   # Parsing and extraction utilities
│   ├── verification.py                      # Post-translation verification
│   └── README.md
│
├── exploration/                             # Analysis and disagreement exploration
│   ├── explore_confidence_within_variant.py # Step 1: within-variant LLM agreement
│   ├── explore_confidence_across_variants.py# Step 2: cross-variant prompt robustness
│   ├── explore_disagreements.py             # Step 3: disagreement typology classification
│   ├── build_disagreement_explorer_data.py  # Builds CSV for the HTML disagreement explorer
│   └── README.md
│
├── utils.py                                 # Shared utilities
└── README.md
```

## Language List

The pipeline uses `generate_language_codes.py` as the single source of truth for the
translation target language list. It combines Unicode CLDR, LOC ISO 639-2, Wikimedia,
and ISO 639-5 into `language_codes_comprehensive.csv`, then `load_language_codes()`
filters to the 270 languages with active Wikipedia projects.

This replaces the old `iso_639_choices.csv` / `iso_639_extended_choices.csv` /
`iso_639_choices_directionality_wikimedia.csv` approach and the `get_directionality()`
function in `data_processing.py`.

## Import Pattern

Scripts in `experiment/` add both the repo root and `scripts/` to `sys.path`:

```python
import sys, os
script_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', '..')))  # repo root
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))        # scripts/

from generate_language_codes import load_language_codes
from scripts.utils import get_data_directory_path
```
