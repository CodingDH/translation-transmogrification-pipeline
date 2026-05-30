#!/usr/bin/env python3
"""
Translation generation with multiple prompt variants for comparative testing.

Four variants, run in this fixed order:
  1. minimal          - bare instruction, no system prompt (baseline)
  2. github_searcher  - frames translation as building a multilingual GitHub search corpus
  3. fluent_speaker - rationale written in the target language
  4. judge            - runs LAST; receives all unique translations from variants 1-3 and all
                        direct services, deduplicated by value, and synthesises the best answer

The judge variant depends on the other three having run first. It can be selected alone only if
prior output files already exist on disk; the script warns otherwise.

Non-LLM sources — MT baselines (GT, EasyNMT, Lingvanex) and the Wikipedia community
reference — are prompt-invariant and loaded from cache for every variant after the
first run. (Ollama-first-pass outputs are also cached the same way.)
"""

import os
import sys
import json
from collections import defaultdict
from typing import List, Optional

import pandas as pd
from rich.console import Console

script_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
scripts_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, scripts_root)
sys.path.insert(0, repo_root)

from generate_translations import generate_translated_terms
from generate_language_codes import load_language_codes
from translation_prompts import (
	PROMPT_VARIANTS,
	PROMPT_DESCRIPTIONS,
	NON_JUDGE_VARIANTS,
	LLM_SERVICE_LABELS,
	DIRECT_SERVICE_LABELS,
)
from scripts.utils import get_data_directory_path

console = Console()

# ── Aggregation for judge ────────────────────────────────────────────────────

def aggregate_variant_translations(
	data_directory_path: str,
	target_terms: List[str],
) -> dict:
	"""
	After all non-judge variants have run, load every output file and build
	per-(language_code, term_source) unique-translation lists for the judge.

	Returns
	-------
	dict
		{term_source: {language_code: {'unique_translations': [...], 'missing_sources': [...]}}}

	Each 'unique_translations' entry is {'translation': str, 'sources': [str]},
	sorted by number of agreeing sources descending so consensus floats to the top.
	"""
	# Accumulate: {(term_source, language_code): {translation_str: [source_label, ...]}}
	contexts: dict = defaultdict(lambda: defaultdict(list))
	missing:  dict = defaultdict(list)

	for term in target_terms:
		term_slug = term.lower().replace(' ', '_')
		prompt_dir  = os.path.join(data_directory_path, 'translated_terms', term_slug, 'prompt_services')
		direct_dir  = os.path.join(data_directory_path, 'translated_terms', term_slug, 'direct_services')

		# LLM variant outputs
		for variant in NON_JUDGE_VARIANTS:
			for svc_key, svc_label in LLM_SERVICE_LABELS.items():
				fpath = os.path.join(prompt_dir, f"{svc_key}_{variant}_translations.csv")
				if not os.path.exists(fpath):
					continue
				col = f"{svc_key}_translated_term"
				try:
					df = pd.read_csv(fpath)
				except Exception:
					continue
				if col not in df.columns:
					continue
				source_label = f"{variant} — {svc_label}"
				for _, row in df.iterrows():
					key = (term, str(row.get('language_code', '')))
					val = row.get(col)
					if pd.notna(val) and str(val).strip() not in ('', 'nan', 'None'):
						contexts[key][str(val).strip()].append(source_label)
					else:
						missing[key].append(source_label)

		# Direct service outputs
		direct_files = {
			'gt_translations.csv':        ('gt_translated_term',        'Google Translate'),
			'enmt_translations.csv':       ('enmt_translated_term',       'EasyNMT'),
			'lingvanex_translations.csv':  ('lingvanex_translated_term',  'Lingvanex'),
			'wikipedia_translations.csv':  ('wikipedia_translated_term',  'Wikipedia'),
		}
		for fname, (col, label) in direct_files.items():
			fpath = os.path.join(direct_dir, fname)
			if not os.path.exists(fpath):
				continue
			try:
				df = pd.read_csv(fpath)
			except Exception:
				continue
			if col not in df.columns:
				continue
			for _, row in df.iterrows():
				key = (term, str(row.get('language_code', '')))
				val = row.get(col)
				if pd.notna(val) and str(val).strip() not in ('', 'nan', 'None'):
					contexts[key][str(val).strip()].append(label)
				else:
					missing[key].append(label)

	# Build the nested result dict
	result: dict = {}
	for (term_source, language_code), translation_sources in contexts.items():
		unique_list = [
			{'translation': t, 'sources': srcs}
			for t, srcs in translation_sources.items()
		]
		unique_list.sort(key=lambda x: len(x['sources']), reverse=True)
		result.setdefault(term_source, {})[language_code] = {
			'unique_translations': unique_list,
			'missing_sources': list(dict.fromkeys(missing[(term_source, language_code)])),
		}

	n_contexts = sum(len(v) for v in result.values())
	console.print(f"Judge context built: {n_contexts} (language, term) pairs", style="bold cyan")
	return result


# ── Per-variant runner ───────────────────────────────────────────────────────

def run_prompt_variant_pipeline(
	variant: str,
	target_terms: List[str],
	data_directory_path: str,
	directionality_df,
	use_cached_translations: bool = True,
	exclude_previous_errors: bool = True,
	gemini_translate: bool = True,
	lingvanex_translate: bool = True,
	deepseek_translate: bool = True,
	gemma_translate: bool = True,
	qwen_translate: bool = True,
	mistral_translate: bool = True,
	term_contexts: dict = None,
) -> Optional[pd.DataFrame]:
	"""
	Run the translation pipeline for one prompt variant.

	For the 'judge' variant, term_contexts must be the dict returned by
	aggregate_variant_translations() — {term_source: {language_code: ctx}}.
	"""
	console.print(f"\n{'='*80}", style="bold cyan")
	console.print(f"PROMPT VARIANT: {variant.upper()}", style="bold yellow")
	console.print(f"Description: {PROMPT_DESCRIPTIONS[variant]}", style="cyan")
	console.print(f"{'='*80}\n", style="bold cyan")

	try:
		combined_translated_df, _, _ = generate_translated_terms(
			data_directory_path=data_directory_path,
			target_terms=target_terms,
			directionality_df=directionality_df,
			gt_translate=True,
			enmt_translate=True,
			openai_translate=True,
			claude_translate=True,
			ollama_translate=True,
			wikipedia_translate=True,
			override_wikipedia=True,
			cached_translations=use_cached_translations,
			return_full_terms_data=False,
			excluding_previous_errors=exclude_previous_errors,
			prompt_variant=variant,
			gemini_translate=gemini_translate,
			lingvanex_translate=lingvanex_translate,
			deepseek_translate=deepseek_translate,
			gemma_translate=gemma_translate,
			qwen_translate=qwen_translate,
			mistral_translate=mistral_translate,
			term_contexts=term_contexts or {},
		)
		console.print(f"\n✓ Completed {variant} variant", style="bold green")
		return combined_translated_df

	except Exception as e:
		console.print(f"\n✗ Error in {variant} variant: {e}", style="bold red")
		raise


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_translation_prompt_variants(target_terms: List[str]):
	"""Orchestrate all prompt variant runs, ensuring judge always runs last."""

	local_data_directory_path = get_data_directory_path()
	should_use_cached_translations  = True
	should_exclude_previous_errors  = True
	should_use_gemini_translate     = True
	should_use_lingvanex_translate  = True
	should_use_deepseek_translate   = True
	should_use_gemma_translate      = True
	should_use_qwen_translate       = True
	should_use_mistral_translate    = True

	existing_directionality_df = load_language_codes()

	console.print(f"\n{'='*80}", style="bold cyan")
	console.print("PROMPT VARIANT TESTING", style="bold yellow")
	console.print(f"{'='*80}", style="bold cyan")
	console.print(f"Target terms: {target_terms}", style="cyan")
	console.print(f"Data directory: {local_data_directory_path}", style="cyan")
	console.print(f"Variants (in run order): {list(PROMPT_VARIANTS.keys())}", style="cyan")
	console.print(f"\n{'='*80}\n", style="bold cyan")

	non_judge = NON_JUDGE_VARIANTS
	all_variants = list(PROMPT_VARIANTS.keys())  # judge is last by dict order

	console.print("Which variants would you like to test?", style="bold yellow")
	for i, v in enumerate(all_variants, 1):
		suffix = "  [requires prior variants on disk]" if v == 'judge' else ""
		console.print(f"  {i}. {v:20s} - {PROMPT_DESCRIPTIONS[v]}{suffix}", style="cyan")
	console.print(f"  {len(all_variants)+1}. ALL (run all variants in order)", style="cyan")
	console.print(f"  {len(all_variants)+2}. QUIT", style="cyan")

	choice = input(f"\nEnter choice (1-{len(all_variants)+2}): ").strip()

	if choice == str(len(all_variants) + 2):
		console.print("Exiting...", style="yellow")
		return

	if choice == str(len(all_variants) + 1):
		variants_to_run = all_variants
	else:
		try:
			idx = int(choice) - 1
			if 0 <= idx < len(all_variants):
				variants_to_run = [all_variants[idx]]
			else:
				console.print("Invalid choice", style="bold red")
				return
		except ValueError:
			console.print("Invalid input", style="bold red")
			return

	# Separate judge from the rest; judge always runs after everything else.
	non_judge_to_run = [v for v in variants_to_run if v != 'judge']
	run_judge        = 'judge' in variants_to_run

	# Run non-judge variants
	for variant in non_judge_to_run:
		run_prompt_variant_pipeline(
			variant=variant,
			target_terms=target_terms,
			data_directory_path=local_data_directory_path,
			directionality_df=existing_directionality_df,
			use_cached_translations=should_use_cached_translations,
			exclude_previous_errors=should_exclude_previous_errors,
			gemini_translate=should_use_gemini_translate,
			lingvanex_translate=should_use_lingvanex_translate,
			deepseek_translate=should_use_deepseek_translate,
			gemma_translate=should_use_gemma_translate,
			qwen_translate=should_use_qwen_translate,
			mistral_translate=should_use_mistral_translate,
		)

	# Run judge last, after aggregating all prior outputs
	if run_judge:
		console.print("\nAggregating translations from all prior variants for judge...", style="bold cyan")
		judge_contexts = aggregate_variant_translations(local_data_directory_path, target_terms)
		if not judge_contexts:
			console.print(
				"⚠ No prior variant outputs found. Run the other variants first before judge.",
				style="bold red",
			)
		else:
			run_prompt_variant_pipeline(
				variant='judge',
				target_terms=target_terms,
				data_directory_path=local_data_directory_path,
				directionality_df=existing_directionality_df,
				use_cached_translations=should_use_cached_translations,
				exclude_previous_errors=should_exclude_previous_errors,
				gemini_translate=should_use_gemini_translate,
				lingvanex_translate=should_use_lingvanex_translate,
				deepseek_translate=should_use_deepseek_translate,
				gemma_translate=should_use_gemma_translate,
				qwen_translate=should_use_qwen_translate,
				mistral_translate=should_use_mistral_translate,
				term_contexts=judge_contexts,
			)

	console.print(f"\n{'='*80}", style="bold green")
	console.print("✓ ALL VARIANTS COMPLETED", style="bold green")
	console.print(f"{'='*80}\n", style="bold green")

	console.print("Results location:", style="bold cyan")
	for term in target_terms:
		term_slug = term.lower().replace(" ", "_")
		console.print(
			f"  {os.path.join(local_data_directory_path, 'translated_terms', term_slug, 'prompt_services')}",
			style="cyan",
		)
	console.print(
		"\nNext step: run scripts/exploration/explore_disagreements.py to verify and compare translations.",
		style="bold cyan",
	)


if __name__ == "__main__":
	target_terms = [
		"Digital Humanities",
	]
	run_translation_prompt_variants(target_terms)
