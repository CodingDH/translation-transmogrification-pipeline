# Translation Pipeline Scripts

Organized into generation and evaluation workflows.

## Structure
```
scripts/
  generate/         - Translation generation pipeline
    generate_translations.py (main)
    data_processing.py
    translation_services.py
    verification.py
    generate_translation_prompts.py
  
  evaluate/         - Translation evaluation & analysis
    evaluate_confidence.py
    evaluate_github_alignment.py
    evaluate_meta_judge.py
    evalute_llm_judge.py
  
  translation_prompts.py    - Shared prompt templates
  utils.py                  - Shared utilities
```

## Usage

**Generate translations:**
```bash
cd scripts
python generate_translations.py
```

**Evaluate translations:**
```bash
cd scripts/evaluate
python evaluate_confidence.py --term "Your Term"
```
