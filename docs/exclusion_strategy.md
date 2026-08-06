# Data Quality & Exclusion Strategy

This document describes the pipeline's automated review signal system, the manual review process, and the three-tier exclusion policy that governs which translations enter each stage of downstream analysis.

The core principle is that even the category of **error** is intrinsically difficult to define in this case. A model producing gibberish for a low-resource language tells us something meaningful about AI coverage gaps, even if that term should never end up as a search term. Different analysis questions therefore warrant different exclusion thresholds.

---

## The Automated Review Signal System

Translation data passes through two quality-control stages before any analysis runs.

### Stage 1 — Automated review signals (`automated_review_signals.csv`)

Notebook `02_translation_overview.ipynb` (§1.9) produces one row per language with automated review flags. Counts below are from the committed Digital Humanities run (881 languages total) and cover the core flags used in the exclusion policy; the CSV also includes additional rationale-language and short-translation annotations used for review.

#### Term content errors — the output is broken or absent

| Flag | Languages | % | Top offending services | What it detects |
|---|---|---|---|---|
| `has_mixed_script` | 94 | 10.7% | Qwen (40), Gemma (38), Llama (19) | Scripts interleaved mid-word with no structural delimiter (Pattern B in `translation_classifier.py`). The term is not salvageable. |
| `has_placeholder_term` | 28 | 3.2% | OpenAI (12), Gemini (11), Claude/Mistral (5 each) | Model returned a refusal string instead of a translation: `"untranslatable"`, `"no direct translation"`, `"Note: ..."`, `"cannot be reliably translated"`, etc. |
| `has_repetition_loop` | 14 | 1.6% | Gemma (7), DeepSeek (3), Mistral (2) | A single whitespace token repeats ≥4 times and represents ≥30% of all tokens. Catches hallucination loops and repeated-token failures. |
| `has_extreme_term_length` | 21 | 2.4% | Mistral (9), Gemini (7), Gemma (5) | Term exceeds 100 characters. Catches two distinct failure modes: (a) hallucination loops long enough to escape word-count detection, and (b) LLM disclaimer text that slipped past placeholder detection. |
| `has_unicode_escape` | 10 | 1.1% | Llama (9), Qwen (1) | Term contains literal `\uXXXX` escape sequences instead of rendered Unicode characters. The model emitted raw JSON encoding rather than text. |

**These five flags represent unambiguous data errors.** The term is either unsalvageable noise, a literal refusal, or a malformed string. The only real decision is whether to drop the language entirely if *all* services trigger the flag, or to retain it with the flagged service excluded.

> **How `has_mixed_script` and `has_placeholder_term` are generated:** Both come from `curate_translation()` in `scripts/exploration/translation_classifier.py`. That function classifies each term and returns an `action` string; notebook 02 §1.9 routes `action='nulled'` → `has_mixed_script` and `action='placeholder'` → `has_placeholder_term`. The five patterns documented in `translation_classifier.py` (A–E) feed into this: **Pattern B** produces `'nulled'`; the placeholder regex produces `'placeholder'`. See *Code Locations* at the end of this document for the full function-to-flag map.

#### Term content concerns — the output is suspicious but potentially valid

| Flag | Languages | % | Top offending services | What it detects |
|---|---|---|---|---|
| `has_source_term` | 420 | 47.7% | OpenAI (275), Claude (187), DeepSeek (144) | Any non-English translation contains the untranslated English source term (`"Digital Humanities"`) or its abbreviation (`"DH"`) as a standalone token. **This is the most ambiguous flag**: borrowing is a genuine finding (e.g. `"Sahtu Digital Humanities"` is a legitimate Dene nativisation), but many cases are simply failures to translate. |
| `has_romanization` | 89 | 10.1% | Gemma (62), Mistral (24), Llama (17) | Model appended an unrequested Latin romanization or source-term prefix alongside the native-script term. The classifier strips it and proposes the curated form in `{svc}_term_clean_{variant}` columns. Despite the name, this flag fires for all four stripping patterns (A, C, D, E) — not only romanization parentheticals (A) but also colon-separated (C), whitespace-separated (D), and equals-separated (E) source prefixes. |
| `has_any_mixing` | 236 | 26.8% | Gemma (123), Qwen (95), Llama (50) | Any secondary-script characters are present in the translation, **regardless of whether they cross the exclusion threshold**. This is a strict superset of `has_mixed_script` + `has_romanization`; it additionally captures sub-threshold cases where the minority script is too sparse to trigger exclusion. Never an exclusion flag — intended purely as a data signal for downstream mixing analysis. See `script_mix_detail()` in `translation_classifier.py` for the raw metrics. |

#### Rationale quality issues

| Flag | Languages | % | Top offending services | What it detects |
|---|---|---|---|---|
| `has_missing_rationale` | 407 | 46.2% | Gemma (221), OpenAI (134), DeepSeek (121) | Raw CSV had a translation with no rationale (or a rationale with no translation). **Already handled automatically**: `enforce_translation_rationale_pairing` nulls the unpaired side at load time, so mismatched cells never enter confidence or disagreement scoring. The flag documents that the original data had a mismatch; it does not mean the language is missing data in downstream analysis. |

#### Cross-service structural disagreements

| Flag | Languages | % | Top offending services | What it detects |
|---|---|---|---|---|
| `has_script_disagreement` | 376 | 42.7% | Llama (235), Qwen (209), Gemma (197) | Services used different writing scripts for the same language. **Not an error** — it is a signal. Services may be making legitimately different choices (academic reconstruction vs. modern adapted script) or one may be wrong. This flag is the primary input to the disagreement typology analysis. |

---

### Stage 2 — Manual exclusions (`manual_exclusions.csv`)

After automated flagging, a manual review pass using `html_files/review_explorer.html` produces `manual_exclusions.csv`. **Generated by:** the "Download Exclusions" button in the explorer. **Consumed by:** `scripts/utils.filter_for_analysis()` — see *Analysis tiers and the `filter_for_analysis` function* below.

This file has one row per *(language × service)* pair and records exclusions at two levels of granularity:

| Column | Meaning |
|---|---|
| `exclude_translation = true` | The service's output for this language is a **model failure**, not a translation |
| `exclude_term_{variant} = true` | This service's output for one specific prompt variant is problematic for search or quality comparison, but the service can translate the language |

#### Analysis tiers and the `filter_for_analysis` function

The two exclusion levels correspond to two conceptually distinct problems, and they warrant different treatment in downstream analysis. The `filter_for_analysis()` function in `scripts/utils.py` exposes this via a `mode` argument:

| CSV column | Conceptual tier | `mode="full"` | `mode="quality"` | `mode="search_ready"` |
|---|---|---|---|---|
| `exclude_translation = true` | **`likely_error`** — output is not a translation (loop, structured tokens, garbage) | retained | **nulled** | **nulled** |
| `exclude_term_{variant} = true` only | **`likely_search_complication`** — output is a valid translation but problematic for search (wrong script, romanisation, too short) | retained | retained | **nulled** |
| neither | `keep` | retained | retained | retained |

The key principle: **errors are data**. A model that produces `[MAN-MADE OBJECT] [BOOK] [COMPUTER]` for Blissymbols is telling you something real about AI coverage gaps. That record should appear in error-rate and coverage analysis (`mode="full"`) but should not skew quality comparisons (`mode="quality"`) or contaminate a search index (`mode="search_ready"`). Nothing is ever dropped from the dataset — only specific service-translation cells are nulled.

```python
from scripts.utils import filter_for_analysis

# error rates, coverage stats — nothing excluded
df_full   = filter_for_analysis(all_dfs[term], term, DATA_DIR, mode="full")

# model quality comparison — likely_error cells nulled
df_quality = filter_for_analysis(all_dfs[term], term, DATA_DIR, mode="quality")

# building the search resource — both tiers nulled
df_search  = filter_for_analysis(all_dfs[term], term, DATA_DIR, mode="search_ready")
```

#### Review decision guide

When working in `review_explorer.html`, use the following heuristics to decide between **Xall** and **Xsrch**:

**Click Xall (→ `likely_error`) when the output is not a translation:**

- Repetition loop: `[MAN-MADE OBJECT] [BOOK] [COMPUTER]`, `digital_(class)+knowledge_(class)`, `kàlā kàlā kàlā…`
- Structured knowledge-graph tokens instead of natural language
- Unicode garbage, raw `\uXXXX` escape sequences
- Empty or whitespace-only output that slipped through automated detection
- A refusal string (`"untranslatable"`, `"no direct equivalent"`) paired with nothing else
- You genuinely cannot tell what language the output is in

**Click Xsrch (→ `likely_search_complication`) when the output is a real translation but has a specific problem:**

- Correct target-language term but appended with English parenthetical or source-language prefix
- Right language, wrong script (e.g. Cyrillic when Arabic expected) — keep for quality analysis, not for search
- Technically valid but you have a better alternative from another service for this variant
- Translation includes the English source term as a component but the rest is a genuine attempt

**Leave both unchecked when:**

- The translation looks reasonable, even if imperfect
- You are uncertain — conservative exclusion is correct for search_ready but not for quality analysis, and the automated flags already handle the clearest cases

From the Digital Humanities review: **42 `likely_error` records** and **113 `likely_search_complication` records** across 352 total review rows.

#### Manual-only patterns (no automated flag)

These categories emerged from review and have no automated detector:

| Pattern | Count (approx.) | Description |
|---|---|---|
| Unintelligible rationale content | ~30 | Rationale text exists and is long enough to pass length checks, but the content is gibberish, in the wrong language, or makes no semantic sense. Concentrated in Ollama on low-resource/ancient languages. Not automatable without LLM-as-judge quality scoring. |
| Search-unsafe characters | ~8 | Terms contain characters that break regex search: ASL gloss notation with brackets and handshape descriptions, Linear A / Meroitic undisplayable codepoints, Blissymbol numeric codes. Technically valid Unicode, but destructive downstream. |
| Constructed/speculative reconstructions | ~4 | Blissymbols (`zbl`), Linear A (`lab`), Elamite (`elx`), Meroitic (`xmr`) — models produce speculative academic reconstructions or outright nonsense for scripts with no real transliteration target. All services excluded for these languages in search-term generation. |
| Single-term refusal with no rationale | ~5 | Model gives a one-word transliteration as the "term" alongside a rationale that just says "not directly translatable." The term is formally valid but methodologically indefensible. |

---

## Three-Tier Exclusion Policy

### Tier 1 — Service & Prompt Exploration (Notebooks 03–04)

**Goal:** Understand how services and prompt variants behave, including their failure modes. **Errors are data here.**

**Philosophy:** A service that hallucinates a repetition loop for Vietnamese is telling you something real about its coverage of Southeast Asian languages. An EasyNMT Lozi output of 109 nonsense characters documents a system limitation that is worth reporting.

| Flag / Exclusion | Policy | Rationale |
|---|---|---|
| `has_missing_rationale` | Retain, show as annotation | Already handled by pairing; the flag itself is a measurement of service reliability |
| `has_mixed_script` | Retain, show as annotation | Documents model failure on a specific script family |
| `has_placeholder_term` | Retain, show as annotation | Documents refusal rate by service and language |
| `has_repetition_loop` | Retain, show as annotation | Documents hallucination rate |
| `has_extreme_term_length` | Retain, show as annotation | Documents verbosity/disclaimer behavior |
| `has_unicode_escape` | Retain, show as annotation | Documents encoding failure rate |
| `has_source_term` | Retain, show as annotation | Borrowing vs. failure — the distinction is itself a finding |
| `has_romanization` | Use curated form | `{svc}_term_clean_{variant}` is available; romanization strip is automatic |
| `has_any_mixing` | Retain as data | Sub-threshold mixing is an analytical finding, not an error; exclusion-worthy cases are already caught by `has_mixed_script` and `has_romanization` |
| `has_script_disagreement` | Retain, show as annotation | Core measurement of service divergence |
| `exclude_translation=true` (manual) | Apply | Whole-service drops for language-service pairs that produce output structurally incompatible with analysis (e.g. EasyNMT hallucination loops, Google Translate for Samoan returning gibberish) |
| `exclude_term_{variant}=true` (manual) | Do not apply | Variant-level quality is exactly what this tier measures |
| `exclude_rationale_{variant}=true` (manual) | Do not apply | Rationale quality is part of the service evaluation |

**Implementing:** Load data with `load_variant_df` (pairing is automatic). Merge in `automated_review_signals.csv` as annotation columns. Apply only `exclude_translation=True` rows from `manual_exclusions.csv`.

---

### Tier 2 — Translation & Rationale Analysis (Notebooks 05–07)

**Goal:** Understand the translation landscape — what terms exist, where services agree, what the disagreement typology is. **Validity of individual translations matters.**

**Philosophy:** A mixed-script garbage term should not count as evidence of a translation. But before excluding, it is worth showing the *differential* — how the distribution changes with and without flagged rows — to demonstrate that exclusions are principled rather than cherry-picked.

| Flag / Exclusion | Policy | Rationale |
|---|---|---|
| `has_missing_rationale` | Already excluded by pairing | No action needed |
| `has_mixed_script` | Exclude term | Unsalvageable; counts as no translation for this service |
| `has_placeholder_term` | Exclude term | Not a translation |
| `has_repetition_loop` | Exclude term | Not a translation |
| `has_extreme_term_length` | Exclude term | Likely a disclaimer or loop, not a translation |
| `has_unicode_escape` | Exclude term | Unrenderable; not usable |
| `has_source_term` | Keep, but flag | Borrowing is a genuine translation strategy; treat as a distinct category in the typology (`STRUCTURAL_ABSENCE` / loan) |
| `has_romanization` | Use curated form | Strip and use `{svc}_term_clean_{variant}` |
| `has_any_mixing` | Retain as data | Sub-threshold mixing is a signal worth analysing (script contact, orthographic variation); use `script_mix_detail()` for mixing degree and minority-script identity |
| `has_script_disagreement` | Keep, categorise | Core input to disagreement typology |
| All manual `exclude_translation=true` | Apply | Human-reviewed hard drops |
| All manual `exclude_term_{variant}=true` | Apply | Human-reviewed variant drops |
| All manual `exclude_rationale_{variant}=true` | Apply for rationale analysis | Rationale excluded; corresponding term may still count if separately valid |

**Implementing:** Pass `--exclusions datasets/.../evaluation/manual_exclusions.csv` to `explore_disagreements.py`. In notebooks, filter out rows where the relevant automated review signal is `True` and the exclusion file confirms the drop. Maintain a parallel "pre-exclusion" table in figures to show the differential.

---

### Tier 3 — Search Term Generation (downstream pipeline)

**Goal:** Produce a clean list of translation strings for querying corpora. **Any noise here propagates into search results.**

**Philosophy:** The cost of a false positive (a bad term matching irrelevant documents) is higher than the cost of a false negative (missing a valid term). When in doubt, exclude.

| Flag / Exclusion | Policy | Rationale |
|---|---|---|
| `has_missing_rationale` | Already excluded | No action needed |
| `has_mixed_script` | Exclude | Would produce broken regex patterns |
| `has_placeholder_term` | Exclude | Not a term |
| `has_repetition_loop` | Exclude | Not a term |
| `has_extreme_term_length` | Exclude | Too long to be a useful search string |
| `has_unicode_escape` | Exclude | Unrenderable; search would fail |
| `has_source_term` | Exclude (non-English) | Searching for "Digital Humanities" in a non-English corpus is already handled by the English term list; duplicate leakage adds noise |
| `has_romanization` | Use curated form only | Raw form may match false positives via mixed-script noise |
| `has_any_mixing` | Retain as annotation | Exclusion-worthy mixing is already handled by `has_mixed_script`; sub-threshold cases are safe to search with |
| `has_script_disagreement` | Use consensus term only | When services disagree on script, use only the majority-script term to avoid cross-script false matches |
| All manual `exclude_translation=true` | Apply | |
| All manual `exclude_term_{variant}=true` | Apply | |
| All manual `exclude_rationale_{variant}=true` | Apply for any rationale-dependent ranking | |
| Search-unsafe characters (manual) | Exclude | ASL gloss notation, Linear A codepoints, Blissymbol codes will break regex search |
| Constructed scripts (`zbl`, `lab`, `elx`, `xmr`) | Exclude entirely | No real-world corpus to search |
| **Uncorroborated singleton** | Exclude | Term produced only once, in an LLM cell with missing/placeholder rationale, and no other source agrees. Treats unexplained singletons as low-confidence and likely junk. |
| Analysis-excluded languages (manual `analysis_exclusion`) | Exclude entirely | Language-level decision applies at Tier 2 and beyond |

**Implementing:** Build the search term list from `disagreement_analysis.csv`. Apply all manual exclusions (`analysis_exclusion`, `search_exclusion`, `term_correction` from `manual_exclusions.csv`). Apply the automated-review-signal based `(language, service)` exclusions (`has_repetition_loop`, `has_unicode_escape`, `has_source_term` non-English, `has_mixed_script`, `has_placeholder_term`, `has_extreme_term_length`). Apply `curate_translation()` to strip parenthetical romanizations. Drop terms with source-term leakage, control characters, or constructed-script languages. **Final pass: apply the corroboration check** — drop any term that lacks both rationale and cross-source agreement (see next subsection).

### The corroboration rule — uncorroborated-singleton exclusion

A search term is kept only if it satisfies **at least one** of two corroboration conditions, evaluated at the **term level** (not per language):

1. **Rationale corroboration**: at least one cell anywhere in the pipeline produced this term with a valid rationale (non-empty, not a placeholder like `"No rationale provided"` / `"N/A"`). The model explained itself for this exact term, in any language.
2. **Cross-source corroboration**: the term was produced by ≥2 distinct cells anywhere — counting any combination of LLM-variant cells across any languages and direct-service cells (Wikipedia, Google Translate, EasyNMT, Lingvanex). Multiple independent sources arrived at the same string, regardless of which language(s) they were translating into.

A term that fails *both* conditions is an **uncorroborated singleton**: produced once anywhere in the pipeline, by a source that gave no explanation. These are auto-excluded from search.

**Note on scope**: corroboration is at the *term level*, not per-(language, term). A term that appears in multiple languages or multiple services anywhere in the pipeline is considered corroborated, even if each individual (language, term) pair on its own is a singleton. This is the more expansive policy: the cost asymmetry of dropping a real translation (missing relevant content from a language community) is higher than including some questionable terms (whose search results can be filtered downstream).

**Worked examples:**

| Scenario | Outcome |
|---|---|
| Mistral/minimal produces `"Foo"` with no rationale (single language); nothing else produces `"Foo"` anywhere | **Drop** — singleton, no rationale, no corroboration anywhere |
| Mistral/minimal (lang A) and Llama/judge (lang B) both produce `"Foo"`, neither with a rationale | **Keep** — two cells produced the term, even across different languages |
| Gemma stamps the same chimera `"घरेलू मानवीयता"` across 4 Indo-Aryan dialects without rationale | **Keep** — 4 cells produced it (this is the trade-off — template-stamping failures get treated as cross-source convergence) |
| Closely-related dialects (Inuktitut ike + ikt) both produce the same syllabics term, no rationale | **Keep** — 2 cells produced it; legitimate dialect convergence preserved |
| Only Wikipedia produces `"Foo"` (no LLM or other source agrees) | **Drop** — single occurrence, no corroborating second source |
| Claude/github_searcher produces `"Foo"` with a valid rationale | **Keep** — rationale alone is sufficient |

**Trade-off**: this rule preserves cross-language dialect convergence at the cost of preserving cross-language template-stamping failures (the Gemma Himalayan-template pattern). The methodological argument: it's better to let questionable terms flow into search and be filtered downstream by hit-count or manual review than to drop legitimate dialect-cluster translations.

**Normalisation**: terms are passed through `curate_translation()` (so parenthetical romanizations don't prevent corroboration: `"Foo (rendering)"` and `"Foo"` agree) and lowercased for the corroboration check (matching GitHub case-insensitive search semantics). Lowercasing was verified to only collapse already-qualifying duplicates — it does not promote any term from "drop" to "keep."

**Implementation**: `_build_corroborated_terms()` in `scripts/generate_search_terms.py`.

---

## Summary Table

| Flag | Tier 1 (Service exploration) | Tier 2 (Translation analysis) | Tier 3 (Search terms) |
|---|---|---|---|
| `has_missing_rationale` | Annotate | Auto-handled by pairing | Auto-handled by pairing |
| `has_mixed_script` | Annotate | Exclude term | Exclude |
| `has_placeholder_term` | Annotate | Exclude term | Exclude |
| `has_repetition_loop` | Annotate | Exclude term | Exclude |
| `has_extreme_term_length` | Annotate | Exclude term | Exclude |
| `has_unicode_escape` | Annotate | Exclude term | Exclude |
| `has_source_term` | Annotate | Keep, categorise | Exclude (non-English) |
| `has_romanization` | Use curated form | Use curated form | Use curated form |
| `has_any_mixing` | Retain as data | Retain as data | Retain as annotation |
| `has_script_disagreement` | Annotate | Keep, categorise | Consensus term only |
| Manual `exclude_translation` | Apply | Apply | Apply |
| Manual `exclude_term_{variant}` | Do not apply | Apply | Apply |
| Manual `exclude_rationale_{variant}` | Do not apply | Apply to rationale use | Apply |
| Search-unsafe / constructed scripts | Annotate | Apply to rationale use | Exclude entirely |
| Uncorroborated singleton (no rationale + no other cell agrees) | Not applicable | Not applicable | Exclude (auto) |
| Manual `analysis_exclusion` (language-level) | Do not apply | Apply (drop language) | Apply (drop language) |

---

## Relationship to Pipeline Stages

```text
notebooks/02_translation_overview.ipynb
    → automated_review_signals.csv           (automated review signals)

html_files/review_explorer.html
    → manual_exclusions.csv       (human review decisions)

notebooks/03_baseline_services.ipynb        [Tier 1 — loose]
notebooks/04_prompt_services.ipynb          [Tier 1 — loose]
    load automated review signals as annotations
    apply only exclude_translation=True from manual_exclusions

notebooks/05_disagreement_analysis.ipynb    [Tier 2 — strict]
notebooks/06_translation_projection.ipynb   [Tier 2 — strict]
notebooks/07_rationale_classification.ipynb [Tier 2 — strict]
notebooks/08_definitional_patterns.ipynb    [Tier 2 — strict]
    apply full manual_exclusions
    exclude flagged term-error flags
    keep source_term and script_disagreement as categories

scripts/generate_search_terms.py            [Tier 3 — strictest]
    apply all exclusions (automated review signals + manual_exclusions.csv)
    apply length / character safety filters
    apply uncorroborated-singleton check (rationale OR ≥2 cells)

notebooks/09_search_results.ipynb           [post-search analysis]
    consume GitHub search results
    measure coverage, language distribution, value of multilingual search
```

---

## Code Locations

### Automated flag implementations

All automated review signals are written to `automated_review_signals.csv` by notebook `02_translation_overview.ipynb` §1.9. The table below shows which function does the actual detection and where that function lives.

| Flag | Detecting function | File |
| --- | --- | --- |
| `has_mixed_script` | `curate_translation()` → `action='nulled'` (Pattern B) | `scripts/exploration/translation_classifier.py` |
| `has_romanization` | `curate_translation()` → `action='stripped'` (Patterns A/C/D/E) | `scripts/exploration/translation_classifier.py` |
| `has_placeholder_term` | `is_placeholder_term()` via `curate_translation()` → `action='placeholder'` | `scripts/exploration/translation_classifier.py` |
| `has_source_term` | `has_source_leakage(text, source_term)` | `scripts/exploration/translation_classifier.py` |
| `has_repetition_loop` | `is_repetition_loop(text)` | `scripts/exploration/translation_classifier.py` |
| `has_extreme_term_length` | `has_extreme_term_length(text, max_chars=100)` | `scripts/exploration/translation_classifier.py` |
| `has_unicode_escape` | `has_unicode_escape(text)` | `scripts/exploration/translation_classifier.py` |
| `has_any_mixing` | `script_mix_detail(text)['any_mixing']` | `scripts/exploration/translation_classifier.py` |
| `has_missing_rationale` | inline `_has_rat()` check on raw prompt-service CSVs | `notebooks/02_translation_overview.ipynb` §1.9 |
| `has_script_disagreement` | `detect_dominant_script()` / `detect_script_disagreement()` | `scripts/utils.py`, called in notebook 02 §1.8.1 |

The `curate_translation()` action → flag routing lives in the automated-review-signals code cell (§1.9): `action='nulled'` → `mixed_by_lang`, `action='stripped'` → `roman_by_lang`, `action='placeholder'` → `placeholder_by_lang`. See `translation_classifier.py` module docstring for the full Pattern → action → flag chain.

### Pairing enforcement (handles `has_missing_rationale` at the data level)

`enforce_translation_rationale_pairing()` in `scripts/utils.py` — called automatically by `load_variant_df()` in `scripts/exploration/explore_confidence_within_variant.py` and by `scripts/exploration/build_review_explorer_data.py`. This function nulls the unpaired side of any translation ↔ rationale mismatch before data reaches any analysis script. The `has_missing_rationale` flag documents what was corrected; it does not mean those languages are missing from downstream analysis.

### Manual exclusions — where they are generated and consumed

| | File / Script |
| --- | --- |
| **Generated by** | `html_files/review_explorer.html` — "Download Exclusions" button exports current session decisions |
| **Consumed by (Step 3)** | `scripts/exploration/explore_disagreements.py` — pass `--exclusions datasets/.../evaluation/manual_exclusions.csv` |
| **Not yet consumed by** | `scripts/exploration/explore_confidence_within_variant.py` (Step 1) and `explore_confidence_across_variants.py` (Step 2) — confidence scores currently reflect raw data including flagged cells |
| **Review interface** | `html_files/review_explorer.html` — load `review_explorer_data.csv`, built by `scripts/exploration/build_review_explorer_data.py` |

---

## Re-running After New Flags

Whenever new automated review signals are added to `translation_classifier.py`, re-run notebook 02 §1.9 to regenerate `automated_review_signals.csv`, then re-run `build_review_explorer_data.py` to refresh `review_explorer_data.csv`. A second human review pass in the HTML explorer is recommended to check whether newly surfaced cases warrant additions to `manual_exclusions.csv`.

The three flags added after the initial manual review (`has_repetition_loop`, `has_extreme_term_length`, `has_unicode_escape`) caught cases that required manual exclusion in the first pass. Re-running notebook 02 will surface any additional languages those flags now auto-detect that were missed in the initial review.
