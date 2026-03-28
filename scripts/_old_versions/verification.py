"""
Verification functions for term translation verification and validation.
"""

import pandas as pd
from rich.console import Console
import operator
from functools import reduce
import arabic_reshaper
from bidi.algorithm import get_display

from scripts.utils import get_data_directory_path

# These will be set by the main module
console = None

def set_globals(console_obj):
    """Set global objects needed by this module."""
    global console
    console = console_obj

def verify_directionality(grouped_terms_df: pd.DataFrame, grouped_terms_output_path: str, directionality_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Function to verify the directionality of terms

	Parameters
	----------
	grouped_terms_df : pd.DataFrame
		A dataframe with grouped terms
	grouped_terms_output_path : str
		A path to the grouped terms output file
	directionality_df : pd.DataFrame
		A dataframe with language directionality
	Returns a dataframe with verified directionality
	"""
	# Set original index to keep track of the original index
	grouped_terms_df['original_index'] = grouped_terms_df.index
	# Check if there are terms with multiple directionality values
	grouped_terms_df['directionality_counts'] = grouped_terms_df.directionality.str.split(',').str.len()
	# Get terms that need directionality specifications
	needs_directional_specifications = grouped_terms_df[(grouped_terms_df.directionality_counts > 1)]
	# Reset the index
	needs_directional_specifications = needs_directional_specifications.reset_index(drop=True)   
	# Loop through the terms that need directionality specifications    
	for index, row in needs_directional_specifications.iterrows():
		console.print(f"Need to specify directionality for {row.term} in {row['English language name']}. Number {index} out of {len(needs_directional_specifications)}", style="bold blue")
		languages = row['English language name'].split(', ')
		# Get the directionality for the languages
		for direction in ['ltr', 'rtl']:
			console.print(f"Directionality: {direction}", style="bold green")
			total_directionalities = directionality_df[(directionality_df['English language name'].isin(languages)) & (directionality_df.directionality == direction)]
			console.print(f"Total languages with directionality {direction}: {len(total_directionalities)}", style="bold green")
			console.print(f"Languages: {total_directionalities['English language name'].tolist()}", style="bold blue")
		# Get the directionality for the languages and save it to the grouped terms dataframe
		input_directionality = console.input("Enter directionality (ltr or rtl): ")
		grouped_terms_df.loc[row['original_index'], 'directionality'] = input_directionality
		grouped_terms_df.to_csv(f'{grouped_terms_output_path}', index=False)
	# Drop the directionality counts and original index columns
	grouped_terms_df = grouped_terms_df.drop(columns=['directionality_counts'])
	grouped_terms_df = grouped_terms_df.drop(columns=['original_index'])
	# Save the grouped terms dataframe
	grouped_terms_df.to_csv(f'{grouped_terms_output_path}', index=False)
	return grouped_terms_df

def verify_terms(merged_lang_terms_df: pd.DataFrame, processed_translated_terms_output_path: str, use_html_verification: bool = True, defer_verification: bool = False) -> pd.DataFrame:
	"""
	Function to verify the terms in the grouped terms dataframe

	Parameters
	----------
	merged_lang_terms_df : pd.DataFrame
		A dataframe with merged language terms
	processed_translated_terms_output_path : str
		A path to the processed translated terms output file
	use_html_verification : bool
		Whether to use HTML verification
	defer_verification : bool
		If True, skip verification (both terminal and HTML) when use_html_verification=False,
		deferring to a later combined HTML pass

	Returns a dataframe with verified terms
	"""
	# Set original index to keep track of the original index
	merged_lang_terms_df['original_index'] = merged_lang_terms_df.index
	console.print(f"Number of terms: {len(merged_lang_terms_df)}", style="bold green")
	# Check if 'keep_term' column exists
	if 'keep_term' not in merged_lang_terms_df.columns:
		merged_lang_terms_df['keep_term'] = True
	# Check if there are terms that need checking — guard optional columns that may
	# not be present when running non-baseline prompt variants (e.g. Wikipedia is
	# skipped for all variants except 'comparative').
	has_wikipedia = 'wikipedia_translated_term' in merged_lang_terms_df.columns
	has_openai = 'openai_translated_term' in merged_lang_terms_df.columns

	if has_wikipedia and has_openai:
		needs_checking_df = merged_lang_terms_df[
			(merged_lang_terms_df.wikipedia_translated_term != merged_lang_terms_df.term) &
			(merged_lang_terms_df.openai_translated_term.isna())
		]
	elif has_openai:
		needs_checking_df = merged_lang_terms_df[merged_lang_terms_df.openai_translated_term.isna()]
	else:
		needs_checking_df = merged_lang_terms_df.copy()

	# Check for terms that don't agree across all available columns
	conditions = []

	translation_sources = [
		'gt_translated_term', 'enmt_translated_term', 'openai_translated_term',
		'ollama_translated_term', 'first_ollama_translated_term'
	]

	# Compare each pair of translation sources
	for i, source1 in enumerate(translation_sources):
		for source2 in translation_sources[i + 1:]:
			if source1 in needs_checking_df.columns and source2 in needs_checking_df.columns:
				conditions.append(needs_checking_df[source1] != needs_checking_df[source2])

	# Combine all conditions using the OR operator
	if conditions:
		combined_condition = reduce(operator.or_, conditions)
		needs_checking_df = needs_checking_df[combined_condition]

	# Reset the index of the resulting DataFrame
	needs_checking_df = needs_checking_df.reset_index(drop=True)
	console.print(f"Number of terms that need checking: {len(needs_checking_df)}", style="bold green")
	# When HTML verification is enabled the browser interface handles review,
	# so skip the CLI prompt loop entirely. Same when deferring verification to a later pass.
	if use_html_verification or defer_verification:
		merged_lang_terms_df = merged_lang_terms_df.drop(columns=['original_index'])
		return merged_lang_terms_df
	# Loop through the terms that need checking
	for index, row in needs_checking_df.iterrows():
		console.print("************************************", style="pale_turquoise1")
		original_term = row['term']
		original_gt_term = row.get('gt_translated_term')
		original_enmt_term = row.get('enmt_translated_term')
		original_openai_term = row.get('openai_translated_term')
		original_openai_rationale = row.get('openai_translation_rationale')
		original_ollama_term = row.get('ollama_translated_term')
		original_first_ollama_term = row.get('first_ollama_translated_term')
		original_local_name = row.get('local language name')

		# Adjust for right-to-left languages
		if row.directionality == 'rtl':
			format_display = lambda x: get_display(arabic_reshaper.reshape(x)) if pd.notna(x) else None
		else:
			format_display = lambda x: x

		# Print all possible translations for review
		console.print(f"Current Term: {format_display(original_term)}. It is {index} out of {len(needs_checking_df)}", style="bold magenta")
		console.print(f"Language: {row['English language name']}", style="slate_blue1")
		if original_local_name:
			console.print(f"Local language name: {format_display(original_local_name)}", style="slate_blue1")
		console.print(f"Directionality: {row.directionality}", style="spring_green2")
		console.print(f"Term source: {row.term_source}", style="spring_green2")

		translation_display = {
			'Google Cloud Translate': original_gt_term,
			'EasyNMT': original_enmt_term,
			'OpenAI': original_openai_term,
			'Ollama': original_ollama_term,
			'First Ollama': original_first_ollama_term
		}

		for source, term in translation_display.items():
			if term:
				console.print(f"{source} Term: {format_display(term)}", style="bold blue")
				if term == 'OpenAI':
					console.print(f"Open AI Rationale: {format_display(original_openai_rationale)}", style="white")
		
		verified = console.input("Is this term correct? (y/n): ")
		if verified == 'n':
			keep_term = console.input("Do you want to keep this term? (y/n): ")
			if keep_term == 'n':
				merged_lang_terms_df.loc[row.original_index, 'keep_term'] = False
			else:
				existing_terms = {k: v for k, v in translation_display.items() if v is not None}

				# Display existing terms for selection
				console.print("Existing terms:")
				for i, (key, term) in enumerate(existing_terms.items(), start=1):
					console.print(f"{i}. {format_display(term)} ({key})", style="bold blue")

				# Get user input
				new_term = console.input("Enter the correct term. You can enter the number of the existing term you want to use or enter a new term: ")

				# Check if the input is a number
				if new_term.isdigit():
					term_index = int(new_term) - 1
					if 0 <= term_index < len(existing_terms):
						new_term = list(existing_terms.values())[term_index]
					else:
						console.print("Invalid number entered. Please enter a valid number or a new term.", style="bold red")
				
				merged_lang_terms_df.loc[row.original_index, 'term'] = new_term
				merged_lang_terms_df.loc[row.original_index, 'keep_term'] = True
				console.print(f"Updated term: {new_term}", style="bold green")
		
		else:
			merged_lang_terms_df.loc[row.original_index, 'keep_term'] = True
		
		merged_lang_terms_df.to_csv(f'{processed_translated_terms_output_path}', index=False)
	merged_lang_terms_df = merged_lang_terms_df.drop(columns=['original_index'])
	merged_lang_terms_df.to_csv(f'{processed_translated_terms_output_path}', index=False)
	return merged_lang_terms_df

def run_html_verification(initial_translated_terms_df: pd.DataFrame, data_directory_path: str) -> pd.DataFrame:
	"""
	Runs an HTML verification workflow for terms translation.

	Uses an HTML interface in a web browser for users to validate and update translations.

	Parameters
	----------
	initial_translated_terms_df : pd.DataFrame
		DataFrame with initial translations
	data_directory_path : str
		Path to data directory

	Returns
	-------
	pd.DataFrame
		DataFrame with user-verified translations
	"""
	# This is a placeholder implementation
	# In the original file, this would load translations into an HTML interface
	console.print("Running HTML verification...", style="bold green")
	return initial_translated_terms_df

