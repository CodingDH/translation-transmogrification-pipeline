#!/usr/bin/env python3
"""
evaluate_llm_judge.py
=====================
Level 2 evaluation: run an LLM judge over all five prompt variant outputs
for each term/language combination. 

The judge scores each variant's translation on a 0–1 scale and explains its
reasoning. Crucially it does NOT pick a single winner — all five scores are
preserved as research data. The disagreements between variants are the
finding, not noise to be resolved.

This is the key architectural difference from WOKIE's LLMConfidenceCalculator,
which collapses to a single output and erases provenance.

Requires:
  - confidence_scores.csv from evaluate_confidence.py (to avoid judging
    rows where primary services already agreed at high confidence)
  - API key for the judge model (Gemini recommended: high limits, low cost,
    top-ranked in Kraus et al. 2025)

Output (data_dir/metadata_files/evaluation/):
  - llm_judge_scores.csv     : per-row judge scores for all variants
  - llm_judge_summary.csv    : aggregate stats
  - judge_disagreements.csv  : cases where judge strongly preferred one variant

Usage:
    python evaluate_llm_judge.py
    python evaluate_llm_judge.py --judge-model gpt-4o --skip-high-confidence
    python evaluate_llm_judge.py --term "Digital Humanities" --sample 50
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.progress import track

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generation_scripts.utils import get_data_directory_path, read_csv_file

console = Console()

ALL_VARIANTS = ['comparative', 'minimal', 'expert_persona', 'contextual', 'native_rationale']

# Variant column names in the scored CSV — when data comes from the combined
# all-variant DataFrame produced by generate_translations_prompts.py, each
# row has a prompt_variant column; when loading per-variant CSVs we merge.
LLM_TERM_COLS = {
    'openai': 'openai_translated_term',
    'claude': 'claude_translated_term',
    'gemini': 'gemini_translated_term',
    'ollama': 'ollama_translated_term',
}

JUDGE_SYSTEM_PROMPT = """You are an expert multilingual translation evaluator with deep knowledge of academic terminology across many languages and scholarly communities.

Your task: evaluate the quality of AI-generated translations of a specialised academic term, considering whether the translation reflects how the scholarly community in that language actually names this concept.

For each candidate translation, assign a score from 0.0 to 1.0:
  1.0 = Clearly correct; this is the term the scholarly community uses
  0.7 = Probably correct but some uncertainty
  0.5 = Plausible but may not reflect actual community usage
  0.2 = Likely a transliteration or loan word with no community adoption
  0.0 = Wrong, hallucinated, or the concept does not exist in this community

CRITICAL: If the concept likely has no established equivalent in the target language community (i.e. it is an imported English-language concept not yet naturalized), ALL scores should be low (≤ 0.3) and your reasoning should explain this explicitly. Low scores across all candidates are a valid and important finding.

Return ONLY valid JSON:
{
  "evaluations": {
    "VARIANT_NAME": {"score": 0.0, "reasoning": "brief explanation"},
    ...
  },
  "concept_exists_in_community": true or false,
  "community_existence_notes": "optional explanation if concept is absent or contested"
}"""


def build_judge_prompt(
    term_source: str,
    language_name: str,
    language_code: str,
    variant_translations: Dict[str, Optional[str]],
    context: Optional[str] = None,
    primary_candidates: Optional[Dict[str, str]] = None,
) -> str:
    """Build the user-facing judge prompt."""
    candidates = "\n".join(
        f"  - {v}: {t if t else '(no translation produced)'}"
        for v, t in variant_translations.items()
    )
    prompt = (
        f"Source term: '{term_source}'\n"
        f"Target language: {language_name} (ISO 639-1: {language_code})\n\n"
        f"Candidate translations by prompting strategy:\n{candidates}\n"
    )
    if primary_candidates:
        primary_str = ", ".join(f"{k}: '{v}'" for k, v in primary_candidates.items() if v)
        if primary_str:
            prompt += f"\nFor reference, primary (non-LLM) services produced: {primary_str}\n"
    if context:
        prompt += f"\nField context: {context}\n"
    prompt += (
        "\nEvaluate each candidate. Return ONLY the JSON object as specified."
    )
    return prompt


def call_judge_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> str:
    """Route to the correct LLM API based on model name."""
    import apikey

    if model.startswith('gemini'):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=apikey.load("CODING_DH_GEMINI_KEY"))
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=1000,
            ),
        )
        return response.text

    elif model.startswith('gpt') or model.startswith('o'):
        from openai import OpenAI
        client = OpenAI(api_key=apikey.load("CODING_DH_OPENAI_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    elif model.startswith('claude'):
        from anthropic import Anthropic
        client = Anthropic(api_key=apikey.load("CODING_DH_CLAUDE_KEY"))
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    else:
        raise ValueError(f"Unsupported judge model: {model}")


def parse_judge_response(
    response_text: str,
    variant_names: List[str],
) -> dict:
    """Parse judge JSON response. Returns dict with scores, reasoning, metadata."""
    cleaned = re.sub(r'^```(?:json)?\s*', '', response_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*$', '', cleaned.strip())
    try:
        data = json.loads(cleaned)
        evals = data.get('evaluations', {})
        scores, reasoning = {}, {}
        for v in variant_names:
            entry = evals.get(v, {})
            scores[v] = float(entry.get('score', 0.0))
            reasoning[v] = str(entry.get('reasoning', ''))
        best = max(scores, key=scores.get) if scores else None
        scores_sorted = sorted(scores.values(), reverse=True)
        gap = (scores_sorted[0] - scores_sorted[1]) if len(scores_sorted) >= 2 else 0.0
        return {
            'scores': scores,
            'reasoning': reasoning,
            'best_variant': best,
            'best_score': scores.get(best, 0.0) if best else 0.0,
            'score_gap': round(gap, 4),
            'judge_agrees': gap >= 0.2,
            'concept_exists': data.get('concept_exists_in_community', True),
            'existence_notes': data.get('community_existence_notes', ''),
            'parse_error': None,
        }
    except Exception as e:
        return {
            'scores': {}, 'reasoning': {}, 'best_variant': None,
            'best_score': 0.0, 'score_gap': 0.0, 'judge_agrees': False,
            'concept_exists': None, 'existence_notes': '',
            'parse_error': f"{type(e).__name__}: {e} — raw: {response_text[:200]}",
        }


def load_combined_variant_df(
    data_directory_path: str,
    term_slug: str,
    variants: List[str] = ALL_VARIANTS,
) -> Optional[pd.DataFrame]:
    """
    Load and merge translations across all variants into a wide DataFrame.

    One row per language, with columns like:
        comparative_openai_translated_term, minimal_openai_translated_term, ...
    """
    dfs = {}
    for variant in variants:
        if variant == 'comparative':
            path = os.path.join(
                data_directory_path, 'metadata_files', 'translated_terms',
                term_slug, 'initial_translated_terms.csv'
            )
        else:
            path = os.path.join(
                data_directory_path, 'metadata_files', 'translated_terms',
                term_slug, 'prompt_variants', f'{variant}_initial_translated_terms.csv'
            )
        if os.path.exists(path):
            df = read_csv_file(path)
            # Rename LLM columns with variant prefix
            rename = {}
            for svc, col in LLM_TERM_COLS.items():
                if col in df.columns:
                    rename[col] = f'{variant}_{svc}_term'
            df = df.rename(columns=rename)
            dfs[variant] = df

    if not dfs:
        return None

    # Merge all variants on language_code + term_source
    base_cols = ['language_code', 'language_name', 'term_source']
    # Also keep primary service columns from comparative (baseline)
    primary_cols = ['gt_translated_term', 'enmt_translated_term', 'wikipedia_translated_term']
    merged = None
    for variant, df in dfs.items():
        keep = [c for c in base_cols if c in df.columns]
        variant_cols = [c for c in df.columns if c.startswith(f'{variant}_')]
        if variant == 'comparative':
            extra = [c for c in primary_cols if c in df.columns]
            keep = keep + extra + variant_cols
        else:
            keep = keep + variant_cols
        df_slim = df[list(dict.fromkeys(keep))].copy()
        if merged is None:
            merged = df_slim
        else:
            merge_on = [c for c in base_cols if c in merged.columns and c in df_slim.columns]
            merged = merged.merge(df_slim, on=merge_on, how='outer')

    return merged


def judge_row(
    row: pd.Series,
    term_source: str,
    variants: List[str],
    judge_model: str,
    context: Optional[str] = None,
    max_retries: int = 2,
) -> dict:
    """Run the LLM judge on a single row."""
    language_name = str(row.get('language_name', row.get('English language name', '')))
    language_code = str(row.get('language_code', ''))

    # Build variant translation dict
    variant_translations = {}
    for v in variants:
        for svc in LLM_TERM_COLS:
            col = f'{v}_{svc}_term'
            val = row.get(col)
            if pd.notna(val) and isinstance(val, str) and val.strip():
                variant_translations[f'{v}_{svc}'] = val.strip()
                break  # use first available LLM service per variant
        else:
            variant_translations[v] = None

    # Primary candidates for reference
    primary = {
        svc: str(row[col]) for svc, col in {
            'GT': 'gt_translated_term',
            'EasyNMT': 'enmt_translated_term',
            'Wikipedia': 'wikipedia_translated_term',
        }.items()
        if col in row.index and pd.notna(row.get(col))
    }

    prompt = build_judge_prompt(
        term_source, language_name, language_code,
        variant_translations, context, primary
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            time.sleep(1)
            response = call_judge_llm(JUDGE_SYSTEM_PROMPT, prompt, judge_model)
            result = parse_judge_response(response, list(variant_translations.keys()))
            result['language_code'] = language_code
            result['language_name'] = language_name
            result['term_source'] = term_source
            result['judge_model'] = judge_model
            result['scores_json'] = json.dumps(result.pop('scores'))
            result['reasoning_json'] = json.dumps(result.pop('reasoning'))
            return result
        except Exception as e:
            last_error = str(e)
            console.print(f"  ⚠ Judge error attempt {attempt+1}: {e}", style="dim yellow")
            time.sleep(2)

    return {
        'language_code': language_code, 'language_name': language_name,
        'term_source': term_source, 'judge_model': judge_model,
        'best_variant': None, 'best_score': 0.0, 'score_gap': 0.0,
        'judge_agrees': False, 'concept_exists': None, 'existence_notes': '',
        'scores_json': '{}', 'reasoning_json': '{}',
        'parse_error': f"Failed after {max_retries+1} attempts: {last_error}",
    }


def run_llm_judge_evaluation(
    data_directory_path: str,
    target_terms: List[str],
    judge_model: str = 'gemini-2.0-flash',
    variants: List[str] = ALL_VARIANTS,
    skip_high_confidence: bool = True,
    confidence_threshold: float = 0.6,
    sample: Optional[int] = None,
    term_contexts: Optional[Dict[str, str]] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run LLM judge over all term/language rows.

    Parameters
    ----------
    skip_high_confidence : bool
        If True, skip rows where primary services already agree above threshold.
        This saves API costs and focuses judge attention on ambiguous cases —
        exactly the cases your paper cares about.
    sample : int, optional
        Randomly sample N rows per term for faster iteration during development.
    """
    all_results = []
    _dir = output_dir or os.path.join(data_directory_path, 'metadata_files', 'evaluation')
    os.makedirs(_dir, exist_ok=True)

    # Optionally load confidence scores to skip high-confidence rows
    conf_path = os.path.join(_dir, 'confidence_scores.csv')
    high_conf_keys = set()
    if skip_high_confidence and os.path.exists(conf_path):
        conf_df = read_csv_file(conf_path)
        high_conf = conf_df[conf_df['primary_above_threshold'] == True]
        high_conf_keys = set(zip(
            high_conf['language_code'].astype(str),
            high_conf['term_source'].astype(str),
        ))
        console.print(
            f"ℹ Skipping {len(high_conf_keys)} high-confidence rows",
            style="dim cyan"
        )

    for term in target_terms:
        term_slug = term.lower().replace(' ', '_')
        context = (term_contexts or {}).get(term)
        console.print(f"\n🔍 Running LLM judge for: {term}", style="bold cyan")
        console.print(f"   Judge model: {judge_model}", style="cyan")

        merged_df = load_combined_variant_df(data_directory_path, term_slug, variants)
        if merged_df is None:
            console.print(f"  ⚠ No data found for '{term}' — skipping", style="bold yellow")
            continue

        # Filter out high-confidence rows
        if high_conf_keys:
            mask = merged_df.apply(
                lambda r: (str(r.get('language_code', '')), str(r.get('term_source', '')))
                not in high_conf_keys,
                axis=1
            )
            to_judge = merged_df[mask]
            console.print(
                f"  {len(to_judge)} rows to judge (of {len(merged_df)} total)",
                style="green"
            )
        else:
            to_judge = merged_df

        # Optional sampling for development
        if sample and len(to_judge) > sample:
            to_judge = to_judge.sample(n=sample, random_state=42)
            console.print(f"  Sampled {sample} rows", style="dim")

        for _, row in track(
            to_judge.iterrows(),
            total=len(to_judge),
            description=f"Judging {term}...",
        ):
            result = judge_row(row, term, variants, judge_model, context)
            all_results.append(result)

    if not all_results:
        console.print("⚠ No judge results produced.", style="bold red")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Summary
    summary = {
        'total_rows_judged': len(results_df),
        'parse_errors': results_df['parse_error'].notna().sum(),
        'judge_agrees_count': results_df['judge_agrees'].sum(),
        'concept_absent_count': (results_df['concept_exists'] == False).sum(),
        'mean_best_score': results_df['best_score'].mean(),
        'variant_wins': results_df['best_variant'].value_counts().to_dict(),
    }
    console.print("\n📊 Judge Summary:", style="bold cyan")
    for k, v in summary.items():
        console.print(f"  {k}: {v}", style="cyan")

    # Disagreements: cases where judge clearly preferred one variant
    disagreements_df = results_df[results_df['judge_agrees'] == True].copy()

    # Save
    judge_path   = os.path.join(_dir, 'llm_judge_scores.csv')
    disagree_path = os.path.join(_dir, 'judge_disagreements.csv')
    results_df.to_csv(judge_path, index=False)
    disagreements_df.to_csv(disagree_path, index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(_dir, 'llm_judge_summary.csv'), index=False)

    console.print(f"\n✓ Saved {len(results_df)} judge results to: {judge_path}", style="bold green")
    console.print(f"  {len(disagreements_df)} clear variant preferences in: {disagree_path}", style="green")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="LLM judge evaluation of prompt variant outputs.")
    parser.add_argument('--term', nargs='+',
                        default=['Computational Humanities', 'Digital Humanities'])
    parser.add_argument('--judge-model', default='gemini-2.0-flash',
                        help='LLM to use as judge (gemini-2.0-flash recommended)')
    parser.add_argument('--variants', nargs='+', default=ALL_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument('--skip-high-confidence', action='store_true', default=True,
                        help='Skip rows where primary services already agreed (saves API costs)')
    parser.add_argument('--no-skip-high-confidence', dest='skip_high_confidence',
                        action='store_false')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N rows per term (for development)')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    data_dir = get_data_directory_path()

    # Term contexts — edit these for your specific terms
    term_contexts = {
        'Digital Humanities': (
            "'Digital Humanities' combines computational methods with the study of "
            "human culture, history, literature, and society. It encompasses text analysis, "
            "digital archiving, and quantitative approaches to humanistic questions. "
            "The term may be borrowed directly into some languages or translated."
        ),
        'Computational Humanities': (
            "'Computational Humanities' applies computational and algorithmic methods "
            "to humanities research questions. It is closely related to Digital Humanities "
            "but emphasizes computational methodology specifically."
        ),
    }

    run_llm_judge_evaluation(
        data_directory_path=data_dir,
        target_terms=args.term,
        judge_model=args.judge_model,
        variants=args.variants,
        skip_high_confidence=args.skip_high_confidence,
        confidence_threshold=args.threshold,
        sample=args.sample,
        term_contexts=term_contexts,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()