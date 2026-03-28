#!/usr/bin/env python3
"""
evaluate_meta_judge.py
======================
Level 3 (experimental): A *different* LLM evaluates the reasoning quality
of the Level 2 judge from evaluate_llm_judge.py.

This tests whether LLMs have stable enough theories of translation to evaluate
each other's reasoning. The expected finding is that meta-judges disagree with
Level 2 judges on lower-resource languages and on concepts with no established
cross-linguistic equivalent — and that breakdown pattern is itself evidence
about the limits of AI translation evaluation.

Design principle: use a DIFFERENT model family from the Level 2 judge to
avoid circular validation. If Level 2 used Gemini, use Claude or GPT-4o here.

Requires: llm_judge_scores.csv from evaluate_llm_judge.py

Output (data_dir/metadata_files/evaluation/):
  - meta_judge_scores.csv       : meta-evaluation of Level 2 reasoning
  - meta_judge_disagreements.csv: cases where meta-judge disagreed with L2
  - judge_chain_summary.csv     : combined L2 + L3 analysis

Usage:
    python evaluate_meta_judge.py
    python evaluate_meta_judge.py --l2-model gemini-2.0-flash --l3-model gpt-4o
    python evaluate_meta_judge.py --sample 30 --focus-absent-concepts
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

META_JUDGE_SYSTEM_PROMPT = """You are an expert in translation evaluation methodology with deep knowledge of multilingualism, language communities, and the politics of scholarly terminology.

You will review assessments made by another AI system (the "Level 2 judge") that evaluated translations of academic terms into various languages.

Your task: evaluate the QUALITY OF THE JUDGE'S REASONING, not the translations themselves.

Assess on these dimensions (0.0–1.0 each):
  reasoning_quality: Is the reasoning coherent and well-supported?
  cultural_sensitivity: Did the judge account for cultural and community context,
    or did it apply generic/Eurocentric assumptions?
  uncertainty_appropriateness: Did the judge express appropriate uncertainty
    for lower-resource languages and absent concepts? (High score = judge was
    appropriately humble; low score = overconfident on uncertain cases)
  internal_consistency: Are the scores consistent with the reasoning?
  absent_concept_detection: If the concept likely has no community equivalent,
    did the judge recognise this? (N/A → 0.5 if concept clearly exists)

Also state:
  l2_judgement_reliable: true/false — would you trust this judge's scores
    to guide a translation decision for this language?
  l2_l3_agreement: true/false — do you broadly agree with the Level 2 judge?
  disagreement_reason: if l2_l3_agreement is false, explain why

Return ONLY valid JSON:
{
  "dimensions": {
    "reasoning_quality": 0.0,
    "cultural_sensitivity": 0.0,
    "uncertainty_appropriateness": 0.0,
    "internal_consistency": 0.0,
    "absent_concept_detection": 0.0
  },
  "overall": 0.0,
  "l2_judgement_reliable": true,
  "l2_l3_agreement": true,
  "disagreement_reason": "",
  "critique": "brief overall assessment"
}"""


def call_meta_judge_llm(system_prompt: str, user_prompt: str, model: str) -> str:
    """Route to appropriate LLM for the meta-judge call."""
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
            temperature=0, max_tokens=1000,
        )
        return response.choices[0].message.content

    elif model.startswith('claude'):
        from anthropic import Anthropic
        client = Anthropic(api_key=apikey.load("CODING_DH_CLAUDE_KEY"))
        message = client.messages.create(
            model=model, max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    else:
        raise ValueError(f"Unsupported meta-judge model: {model}")


def build_meta_judge_prompt(row: pd.Series) -> str:
    """Build the prompt for the meta-judge from a Level 2 result row."""
    scores = {}
    reasoning = {}
    try:
        scores = json.loads(str(row.get('scores_json', '{}')))
        reasoning = json.loads(str(row.get('reasoning_json', '{}')))
    except Exception:
        pass

    scores_str = "\n".join(f"  {v}: {s:.2f}" for v, s in scores.items())
    reasoning_str = "\n".join(f"  {v}: {r}" for v, r in reasoning.items())

    return (
        f"Source term: '{row.get('term_source', '?')}'\n"
        f"Target language: {row.get('language_name', '?')} "
        f"({row.get('language_code', '?')})\n\n"
        f"Level 2 judge model: {row.get('judge_model', '?')}\n"
        f"Level 2 best variant: {row.get('best_variant', '?')} "
        f"(score: {row.get('best_score', '?')})\n"
        f"Level 2 concept_exists assessment: {row.get('concept_exists', '?')}\n"
        f"Level 2 existence notes: {row.get('existence_notes', '')}\n\n"
        f"Level 2 scores by variant:\n{scores_str}\n\n"
        f"Level 2 reasoning by variant:\n{reasoning_str}\n\n"
        "Evaluate the quality and reliability of the Level 2 judge's assessment. "
        "Return ONLY the JSON object as specified."
    )


def parse_meta_judge_response(response_text: str) -> dict:
    """Parse the meta-judge response."""
    cleaned = re.sub(r'^```(?:json)?\s*', '', response_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*$', '', cleaned.strip())
    try:
        data = json.loads(cleaned)
        dims = data.get('dimensions', {})
        return {
            'meta_reasoning_quality': float(dims.get('reasoning_quality', 0.0)),
            'meta_cultural_sensitivity': float(dims.get('cultural_sensitivity', 0.0)),
            'meta_uncertainty_appropriateness': float(dims.get('uncertainty_appropriateness', 0.0)),
            'meta_internal_consistency': float(dims.get('internal_consistency', 0.0)),
            'meta_absent_concept_detection': float(dims.get('absent_concept_detection', 0.0)),
            'meta_overall': float(data.get('overall', 0.0)),
            'l2_judgement_reliable': bool(data.get('l2_judgement_reliable', True)),
            'l2_l3_agreement': bool(data.get('l2_l3_agreement', True)),
            'disagreement_reason': str(data.get('disagreement_reason', '')),
            'meta_critique': str(data.get('critique', '')),
            'meta_parse_error': None,
        }
    except Exception as e:
        return {
            'meta_reasoning_quality': None, 'meta_cultural_sensitivity': None,
            'meta_uncertainty_appropriateness': None, 'meta_internal_consistency': None,
            'meta_absent_concept_detection': None, 'meta_overall': None,
            'l2_judgement_reliable': None, 'l2_l3_agreement': None,
            'disagreement_reason': '', 'meta_critique': '',
            'meta_parse_error': f"{type(e).__name__}: {e} — raw: {response_text[:200]}",
        }


def run_meta_judge_evaluation(
    data_directory_path: str,
    l3_model: str = 'gpt-4o',
    sample: Optional[int] = None,
    focus_absent_concepts: bool = False,
    focus_low_reliability: bool = False,
    output_dir: Optional[str] = None,
    max_retries: int = 2,
) -> pd.DataFrame:
    """
    Run meta-judge evaluation over Level 2 judge results.

    Parameters
    ----------
    focus_absent_concepts : bool
        If True, only evaluate rows where Level 2 judged concept_exists=False.
        These are the most theoretically interesting cases for your paper.
    focus_low_reliability : bool
        If True, only evaluate rows where Level 2 had low best_score (< 0.4).
    """
    _dir = output_dir or os.path.join(data_directory_path, 'metadata_files', 'evaluation')
    judge_path = os.path.join(_dir, 'llm_judge_scores.csv')

    if not os.path.exists(judge_path):
        console.print(
            f"⚠ Level 2 judge scores not found at {judge_path}\n"
            "  Run evaluate_llm_judge.py first.",
            style="bold red"
        )
        return pd.DataFrame()

    judge_df = read_csv_file(judge_path)
    console.print(
        f"Loaded {len(judge_df)} Level 2 judge results", style="cyan"
    )

    # Focus filters
    to_evaluate = judge_df.copy()
    if focus_absent_concepts:
        to_evaluate = to_evaluate[to_evaluate['concept_exists'] == False]
        console.print(
            f"  Focusing on {len(to_evaluate)} absent-concept cases", style="dim cyan"
        )
    if focus_low_reliability:
        to_evaluate = to_evaluate[pd.to_numeric(to_evaluate['best_score'], errors='coerce') < 0.4]
        console.print(
            f"  Focusing on {len(to_evaluate)} low-reliability cases", style="dim cyan"
        )

    if sample and len(to_evaluate) > sample:
        to_evaluate = to_evaluate.sample(n=sample, random_state=42)
        console.print(f"  Sampled {sample} rows for meta-evaluation", style="dim")

    console.print(
        f"\n🔬 Running Level 3 meta-judge ({l3_model}) "
        f"on {len(to_evaluate)} rows...",
        style="bold cyan"
    )

    meta_results = []
    for _, row in track(to_evaluate.iterrows(), total=len(to_evaluate), description="Meta-judging..."):
        prompt = build_meta_judge_prompt(row)
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                time.sleep(1)
                response = call_meta_judge_llm(META_JUDGE_SYSTEM_PROMPT, prompt, l3_model)
                parsed = parse_meta_judge_response(response)
                parsed['language_code'] = row.get('language_code', '')
                parsed['language_name'] = row.get('language_name', '')
                parsed['term_source'] = row.get('term_source', '')
                parsed['l2_judge_model'] = row.get('judge_model', '')
                parsed['l3_judge_model'] = l3_model
                parsed['l2_best_variant'] = row.get('best_variant', '')
                parsed['l2_best_score'] = row.get('best_score', '')
                parsed['l2_concept_exists'] = row.get('concept_exists', '')
                meta_results.append(parsed)
                break
            except Exception as e:
                last_error = str(e)
                console.print(f"  ⚠ Meta-judge error attempt {attempt+1}: {e}", style="dim yellow")
                time.sleep(2)
        else:
            meta_results.append({
                'language_code': row.get('language_code', ''),
                'language_name': row.get('language_name', ''),
                'term_source': row.get('term_source', ''),
                'l2_judge_model': row.get('judge_model', ''),
                'l3_judge_model': l3_model,
                'l2_best_variant': row.get('best_variant', ''),
                'l2_best_score': row.get('best_score', ''),
                'l2_concept_exists': row.get('concept_exists', ''),
                'meta_parse_error': f"Failed: {last_error}",
            })

    meta_df = pd.DataFrame(meta_results)

    # Identify disagreements: L3 disagrees with L2
    disagreements = meta_df[meta_df['l2_l3_agreement'] == False].copy()

    # Chain summary
    n_reliable = meta_df['l2_judgement_reliable'].sum() if 'l2_judgement_reliable' in meta_df else 0
    chain_summary = {
        'l3_model': l3_model,
        'total_meta_evaluated': len(meta_df),
        'l2_l3_agreement_rate': round(meta_df['l2_l3_agreement'].mean(), 4)
            if 'l2_l3_agreement' in meta_df else None,
        'l2_reliable_rate': round(n_reliable / len(meta_df), 4) if len(meta_df) else None,
        'mean_meta_overall': round(meta_df['meta_overall'].mean(), 4)
            if 'meta_overall' in meta_df else None,
        'mean_cultural_sensitivity': round(meta_df['meta_cultural_sensitivity'].mean(), 4)
            if 'meta_cultural_sensitivity' in meta_df else None,
        'l2_l3_disagreements': len(disagreements),
        'parse_errors': meta_df['meta_parse_error'].notna().sum()
            if 'meta_parse_error' in meta_df else 0,
    }

    console.print("\n📊 Meta-Judge Summary:", style="bold cyan")
    for k, v in chain_summary.items():
        console.print(f"  {k}: {v}", style="cyan")

    # Save
    meta_path     = os.path.join(_dir, 'meta_judge_scores.csv')
    disagree_path = os.path.join(_dir, 'meta_judge_disagreements.csv')
    summary_path  = os.path.join(_dir, 'judge_chain_summary.csv')

    meta_df.to_csv(meta_path, index=False)
    disagreements.to_csv(disagree_path, index=False)
    pd.DataFrame([chain_summary]).to_csv(summary_path, index=False)

    console.print(f"\n✓ Meta-judge results saved to: {meta_path}", style="bold green")
    console.print(f"  {len(disagreements)} L2/L3 disagreements: {disagree_path}", style="green")

    return meta_df


def main():
    parser = argparse.ArgumentParser(
        description="Level 3 meta-judge: evaluate quality of Level 2 judge reasoning."
    )
    parser.add_argument('--l3-model', default='gpt-4o',
                        help='Model for meta-judge (should differ from L2 judge model)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Randomly sample N rows from Level 2 results')
    parser.add_argument('--focus-absent-concepts', action='store_true',
                        help='Only meta-judge rows where L2 found concept absent')
    parser.add_argument('--focus-low-reliability', action='store_true',
                        help='Only meta-judge rows where L2 best_score < 0.4')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    data_dir = get_data_directory_path()
    run_meta_judge_evaluation(
        data_directory_path=data_dir,
        l3_model=args.l3_model,
        sample=args.sample,
        focus_absent_concepts=args.focus_absent_concepts,
        focus_low_reliability=args.focus_low_reliability,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()