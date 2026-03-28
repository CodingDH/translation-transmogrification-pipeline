#!/usr/bin/env python3
"""
Translation generation with multiple prompt variants for comparative testing.

This script tests different prompt engineering strategies to understand their impact
on translation quality across multiple LLM services (OpenAI, Claude, Ollama).

Structure:
- Baseline services (Google Translate, EasyNMT, Wikipedia) stay the same
- LLM services tested with 5 different prompt variants:
  1. minimal - bare-bones instruction
  2. comparative - shows existing translations
  3. expert_persona - positions as domain expert
  4. contextual - rich semantic context
  5. native_rationale - rationale in target language

Results are saved separately by variant for easy comparison.
"""

import os
import sys
import json
import urllib.parse
import webbrowser
import pandas as pd
from typing import List, Optional
from rich.console import Console

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_translations import (
	generate_translated_terms,
	get_directionality,
)
from translation_prompts import PROMPT_VARIANTS, PROMPT_DESCRIPTIONS
from utils import get_data_directory_path, read_csv_file

console = Console()


def generate_comparison_data(target_terms: List[str], data_directory_path: str) -> List[dict]:
	"""
	Load and compare translations across all prompt variants.

	Parameters
	----------
	target_terms : List[str]
		The terms that were translated
	data_directory_path : str
		Base data directory path

	Returns
	-------
	List[dict]
		List of comparison records
	"""
	comparison_data = []
	variants = ['minimal', 'comparative', 'expert_persona', 'contextual', 'native_rationale']

	for term in target_terms:
		term_slug = term.lower().replace(" ", "_")
		base_path = os.path.join(
			data_directory_path,
			"metadata_files",
			"translated_terms",
			term_slug
		)

		# Load baseline (comparative) for language info
		baseline_path = os.path.join(base_path, "initial_translated_terms.csv")
		if not os.path.exists(baseline_path):
			console.print(f"⚠ No baseline data for {term}", style="yellow")
			continue

		try:
			baseline_df = read_csv_file(baseline_path)

			# Load variant data with deduplication by coding_dh_date
			variant_dfs = {}
			for variant in variants:
				variant_path = os.path.join(base_path, "prompt_variants", f"{variant}_initial_translated_terms.csv")
				if os.path.exists(variant_path):
					variant_df = read_csv_file(variant_path)
					# Keep only the most recent version per language_code and term_source
					if 'coding_dh_date' in variant_df.columns:
						variant_df['coding_dh_date'] = pd.to_datetime(variant_df['coding_dh_date'], errors='coerce')
						# NaT values sort to the end, so valid dates are preferred automatically
						variant_df = variant_df.sort_values('coding_dh_date', ascending=False).drop_duplicates(
							subset=['language_code', 'term_source'], keep='first'
						)
						# Log if rows had missing/invalid timestamps
						nat_count = variant_df['coding_dh_date'].isna().sum()
						if nat_count > 0:
							console.print(f"⚠ {variant} variant for {term}: {nat_count} rows with missing timestamps (using first occurrence)", style="dim yellow")
					variant_dfs[variant] = variant_df

			# Merge comparison data
			for _, row in baseline_df.iterrows():
				comparison_row = {
					'term_source': row.get('term_source', term),
					'language_code': row.get('language_code', ''),
					'language_name': row.get('language_name', row.get('English language name', '')),
				}

				# Add variant translations
				for variant in variants:
					if variant in variant_dfs:
						variant_df = variant_dfs[variant]
						# Find matching row in variant data
						match = variant_df[
							(variant_df['language_code'] == row['language_code']) &
							(variant_df['term_source'] == row.get('term_source', term))
						]
						if not match.empty:
							comparison_row[f'{variant}_term'] = match.iloc[0].get('term')
						else:
							comparison_row[f'{variant}_term'] = None
					else:
						comparison_row[f'{variant}_term'] = None

				comparison_data.append(comparison_row)

		except Exception as e:
			console.print(f"⚠ Error loading data for {term}: {e}", style="yellow")

	return comparison_data


def open_comparator(comparison_data: List[dict]) -> None:
	"""
	Open the variant comparator HTML in browser.

	Parameters
	----------
	comparison_data : List[dict]
		The comparison data to pass to the HTML
	"""
	# Encode data for URL parameter
	data_json = json.dumps(comparison_data)
	encoded_data = urllib.parse.quote(data_json)

	# Get comparator HTML path
	comparator_path = os.path.join(os.path.dirname(__file__), "..", "files", "translation_comparator.html")
	comparator_path = os.path.abspath(comparator_path)

	if os.path.exists(comparator_path):
		browser_url = f"file://{comparator_path}?data={encoded_data}"
		webbrowser.open(browser_url)
		console.print(f"✓ Opened: {comparator_path}", style="bold green")
	else:
		console.print(f"⚠ Comparator HTML not found at {comparator_path}", style="bold yellow")


def run_prompt_variant_pipeline(
	variant: str,
	target_terms: List[str],
	data_directory_path: str,
	directionality_df,
	skip_cached_files: bool = False,
	use_cached_translations: bool = True,
	exclude_previous_errors: bool = True,
	use_html_verification: bool = True,
	run_all_variants: bool = False,
	defer_verification: bool = False,
	gemini_translate: bool = False,
	lingvanex_translate: bool = False,
	term_contexts: dict = None,
) -> Optional[pd.DataFrame]:
	"""
	Run the translation pipeline for a specific prompt variant.

	Parameters
	----------
	variant : str
		The prompt variant to test ('minimal', 'comparative', 'expert_persona', etc.)
	target_terms : List[str]
		Terms to translate
	data_directory_path : str
		Base data directory
	directionality_df : pd.DataFrame
		Language directionality data
	skip_cached_files : bool
		Whether to skip cached files
	use_cached_translations : bool
		Whether to use cached translations from services
	exclude_previous_errors : bool
		Whether to exclude previously errored translations
	use_html_verification : bool
		Whether to use HTML verification interface
	run_all_variants : bool
		Whether running all variants in sequence
	defer_verification : bool
		If True, skip verification when use_html_verification=False, deferring to later HTML pass
	Returns
	-------
	Optional[pd.DataFrame]
		The combined translated DataFrame, or None if the pipeline errored.
	"""
	console.print(f"\n{'='*80}", style="bold cyan")
	console.print(f"PROMPT VARIANT: {variant.upper()}", style="bold yellow")
	console.print(f"Description: {PROMPT_DESCRIPTIONS[variant]}", style="cyan")
	console.print(f"{'='*80}\n", style="bold cyan")

	# GT, EasyNMT, Wikipedia, and the first Ollama pass are prompt-invariant —
	# their outputs are identical regardless of which LLM prompt is being tested.
	# Run them only for the baseline 'comparative' variant; all other variants
	# load those results from cache (use_cached_translations=True) and only
	# re-run the LLM services whose prompts actually change.
	is_baseline = variant == 'comparative'
	should_use_gt_translate = is_baseline
	should_use_enmt_translate = is_baseline
	should_use_wikipedia_translate = is_baseline
	should_use_ollama_translate = is_baseline  # First Ollama pass is also baseline
	# Lingvanex is a primary (non-LLM) service — prompt-invariant, baseline only
	should_use_lingvanex_translate = is_baseline and lingvanex_translate
	should_use_openai_translate = True
	should_use_claude_translate = True
	# Gemini is an LLM — runs on every variant (prompt choice matters)
	should_use_gemini_translate = gemini_translate
	should_override_wikipedia = True
	# Only run the second Ollama pass for comparative (it depends on first pass output)
	should_run_llama_twice = is_baseline

	try:
		# Run the main translation pipeline
		combined_translated_df, _, _ = generate_translated_terms(
			data_directory_path=data_directory_path,
			target_terms=target_terms,
			directionality_df=directionality_df,
			gt_translate=should_use_gt_translate,
			enmt_translate=should_use_enmt_translate,
			openai_translate=should_use_openai_translate,
			claude_translate=should_use_claude_translate,
			ollama_translate=should_use_ollama_translate,
			wikipedia_translate=should_use_wikipedia_translate,
			override_wikipedia=should_override_wikipedia,
			run_llama_twice=should_run_llama_twice,
			skip_cached_files=skip_cached_files,
			cached_translations=use_cached_translations,
			return_full_terms_data=False,
			excluding_previous_errors=exclude_previous_errors,
			use_html_verification=use_html_verification,
			defer_verification=defer_verification,
			prompt_variant=variant,
			gemini_translate=should_use_gemini_translate,
			lingvanex_translate=should_use_lingvanex_translate,
			term_contexts=term_contexts,
		)

		console.print(f"\n✓ Completed {variant} variant", style="bold green")
		return combined_translated_df

	except Exception as e:
		console.print(f"\n✗ Error in {variant} variant: {e}", style="bold red")
		raise


def run_translation_prompt_variants(target_terms: List[str]):
	"""
	Run prompt variant testing for specified target terms.

	Parameters
	----------
	target_terms : List[str]
		Terms to translate across all prompt variants
	"""

	# Configuration
	local_data_directory_path = get_data_directory_path()
	existing_directionality_path = os.path.join(
		local_data_directory_path, "metadata_files", "iso_639_choices_directionality_wikimedia.csv"
	)

	# Flags
	should_skip_cached_files = False
	should_use_cached_translations = True
	should_exclude_previous_errors = True
	should_use_html_verification = True  # Set to False if not using browser verification
	should_use_gemini_translate = False   # Use Gemini (gemini-2.0-flash) — recommended by Kraus et al. 2025
	should_use_lingvanex_translate = False # Use Lingvanex — top-ranked primary service in Kraus et al. 2025

	# Load directionality
	existing_directionality_df = get_directionality(existing_directionality_path)

	console.print(f"\n{'='*80}", style="bold cyan")
	console.print("COMPARATIVE PROMPT VARIANT TESTING", style="bold yellow")
	console.print(f"{'='*80}", style="bold cyan")
	console.print(f"Target terms: {target_terms}", style="cyan")
	console.print(f"Data directory: {local_data_directory_path}", style="cyan")
	console.print(f"Variants to test: {list(PROMPT_VARIANTS.keys())}", style="cyan")
	console.print(f"\n{'='*80}\n", style="bold cyan")

	# Ask user which variants to run
	console.print("Which variants would you like to test?", style="bold yellow")
	for i, variant in enumerate(PROMPT_VARIANTS.keys(), 1):
		console.print(f"  {i}. {variant:20s} - {PROMPT_DESCRIPTIONS[variant]}", style="cyan")
	console.print(f"  {len(PROMPT_VARIANTS)+1}. ALL (run all variants)", style="cyan")
	console.print(f"  {len(PROMPT_VARIANTS)+2}. QUIT", style="cyan")

	choice = input("\nEnter choice (1-7): ").strip()

	if choice == str(len(PROMPT_VARIANTS) + 2):
		console.print("Exiting...", style="yellow")
		return

	variants_to_run = []

	if choice == str(len(PROMPT_VARIANTS) + 1):
		# Run all
		variants_to_run = list(PROMPT_VARIANTS.keys())
	else:
		try:
			idx = int(choice) - 1
			if 0 <= idx < len(PROMPT_VARIANTS):
				variants_to_run = [list(PROMPT_VARIANTS.keys())[idx]]
			else:
				console.print("Invalid choice", style="bold red")
				return
		except ValueError:
			console.print("Invalid input", style="bold red")
			return

	# Run selected variants — always skip in-loop HTML verification so we can do
	# a single combined verification pass after all variants have finished.
	run_all = len(variants_to_run) > 1
	all_translated_dfs = []
	for variant in variants_to_run:
		result_df = run_prompt_variant_pipeline(
			variant=variant,
			target_terms=target_terms,
			data_directory_path=local_data_directory_path,
			directionality_df=existing_directionality_df,
			skip_cached_files=should_skip_cached_files,
			use_cached_translations=should_use_cached_translations,
			exclude_previous_errors=should_exclude_previous_errors,
			use_html_verification=False,  # Always defer to combined pass below
			run_all_variants=run_all,
			defer_verification=run_all,  # Skip verification in individual variants when running multiple
			gemini_translate=should_use_gemini_translate,
			lingvanex_translate=should_use_lingvanex_translate,
			term_contexts=None,  # Can pass specific contexts if desired
		)
		if result_df is not None:
			result_df['prompt_variant'] = variant
			all_translated_dfs.append(result_df)

	# Single HTML verification pass over all variant results combined
	if should_use_html_verification and all_translated_dfs:
		console.print(f"\n🌐 Opening HTML verification for all {len(variants_to_run)} variant(s)...", style="bold cyan")
		combined_for_verification = pd.concat(all_translated_dfs, ignore_index=True)
		from scripts.generate_translations import run_html_verification
		run_html_verification(combined_for_verification, local_data_directory_path)

	console.print(f"\n{'='*80}", style="bold green")
	console.print("✓ ALL VARIANTS COMPLETED", style="bold green")
	console.print(f"{'='*80}\n", style="bold green")

	# Print summary
	console.print("Results location:", style="bold cyan")
	for term in target_terms:
		term_slug = term.lower().replace(" ", "_")
		variants_dir = os.path.join(
			local_data_directory_path,
			"metadata_files",
			"translated_terms",
			term_slug,
			"prompt_variants"
		)
		console.print(f"  {variants_dir}", style="cyan")

	# Generate and open comparator
	console.print("\n📊 Generating comparison data...", style="bold cyan")
	comparison_data = generate_comparison_data(target_terms, local_data_directory_path)

	if comparison_data:
		console.print(f"✓ Generated comparison data for {len(comparison_data)} translations", style="bold green")
		console.print("\n🌐 Opening Translation Variant Comparator...\n", style="bold cyan")
		open_comparator(comparison_data)
	else:
		console.print("⚠ No comparison data generated", style="bold yellow")


if __name__ == "__main__":
	target_terms = [
		"Computational Humanities",
		# Add more terms as needed
	]
	run_translation_prompt_variants(target_terms)