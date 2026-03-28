# Evaluation Scripts

Multi-level evaluation of translation quality and consistency across services and prompt variants.

## Evaluation Levels

### Level 0: Within-Variant Service Agreement
**`evaluate_confidence_within_variant.py`** ✅
- For each prompt variant separately, measure how much translation services agree
- **Question**: "Do Google Translate, EasyNMT, and Lingvanex produce the same translation?"
- **Output**: Per-variant service agreement scores, low-confidence rows
- **Run this first** — generates baseline confidence metrics for all variants

### Level 1: Cross-Variant Prompt Robustness
**`evaluate_confidence_across_variants.py`** ⏳ (Template ready)
- For each term/language, compare translations across different prompt variants
- **Question**: "Does the 'comparative' variant produce the same translation as 'minimal'?"
- **Output**: Variant consistency scores, divergence detection
- **Run this second** — uses output from Level 0

### Additional Evaluations
- **evaluate_github_alignment.py** - Alignment with GitHub language metadata
- **evaluate_meta_judge.py** - Meta-judge LLM evaluation
- **evalute_llm_judge.py** - LLM-based translation quality judgment

## Recommended Workflow

```
Level 0: evaluate_confidence_within_variant.py
         ↓ (generates confidence_scores.csv per variant)
Level 1: evaluate_confidence_across_variants.py
         ↓ (generates variant_agreement.csv)
Level 2+: Other evaluations (github_alignment, meta_judge, etc.)
```

## Quick Start

```bash
# Level 0: Service agreement within each variant
python evaluate_confidence_within_variant.py --term "Digital Humanities"

# Level 1: Consistency across variants (when ready for implementation)
# python evaluate_confidence_across_variants.py --term "Digital Humanities"
```
