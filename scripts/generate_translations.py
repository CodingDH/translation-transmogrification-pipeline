# Standard library imports
import codecs
import html
import json
import os
import time
import warnings
from typing import List, Tuple, Callable, Optional
import ast
import inspect
import re
import operator
from functools import reduce
import shutil
import datetime


import threading
# Local application/library specific imports
import apikey
# Related third-party imports
import pandas as pd
import requests
import arabic_reshaper
from bidi.algorithm import get_display
from bs4 import BeautifulSoup
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from easynmt import EasyNMT
from transformers import AutoTokenizer
from rich.console import Console
from tqdm import tqdm
from openai import OpenAI, OpenAIError
from anthropic import Anthropic, APIError
from google import genai as google_genai
import wikipediaapi
import ollama
from pydantic import BaseModel, Field, ValidationError
import sys
sys.path.append("..")
from data_generation_scripts.utils import get_data_directory_path, read_csv_file, log_error_to_file, clean_write_error_file
from data_generation_scripts.translation_prompts import get_prompt, PROMPT_VARIANTS, PROMPT_DESCRIPTIONS
warnings.filterwarnings('ignore')
# Load Google Cloud credentials. You can get your own credentials by following the instructions here: https://cloud.google.com/translate/docs/setup and saving them with apikey.save("GOOGLE_TRANSLATE_CREDENTIALS", "path/to/your/credentials.json")
google_translate_key_path = apikey.load("GOOGLE_TRANSLATE_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_file(
	google_translate_key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

translate_client = translate.Client(credentials=credentials)

openai_key_path = apikey.load("CODING_DH_OPENAI_KEY")
openai_project_id = apikey.load("CODING_DH_OPENAI_PROJECT_ID")
openai_organization_id = apikey.load("CODING_DH_OPENAI_ORGANIZATION_ID")
client = OpenAI(
	api_key=openai_key_path,
	project=openai_project_id,
	organization=openai_organization_id
)

# Load Claude API credentials
claude_api_key = apikey.load("CODING_DH_CLAUDE_KEY")
claude_client = Anthropic(api_key=claude_api_key)

# Load Gemini credentials (optional — only used if CODING_DH_GEMINI_KEY is set)
gemini_client = None
GEMINI_MODEL = "gemini-2.0-flash"
try:
	gemini_api_key = apikey.load("CODING_DH_GEMINI_KEY")
	gemini_client = google_genai.Client(api_key=gemini_api_key)
except Exception:
	pass  # Gemini is optional; pipeline runs without it

console = Console()

# Initialize the EasyNMT model once
model = EasyNMT('opus-mt')

# Global variable to store current prompt variant (used by translation functions)
current_prompt_variant = 'comparative'

# Optional per-term context strings for the 'contextual' prompt variant.
# Keys are term_source values, values are plain-text field descriptions.
# Set via term_contexts parameter in generate_translated_terms().
current_term_contexts: dict = {}

MAX_CONSECUTIVE_TIMEOUTS = 5  # Stop after 5 consecutive timeouts


# Import from new modules
from generate.data_processing import (
    TranslationResponse, parse_translation_response, check_detect_language,
    extract_dictionaries_from_string, extract_ollama_translated_term,
    get_directionality, is_enmt_model_available
)
from generate.translation_services import (
    get_gt_translation, get_enmt_translation, get_openai_translation,
    get_claude_translation, get_gemini_translation, get_lingvanex_translation,
    get_ollama_translation, check_if_wikipedia_page_exists
)
from generate.verification import (
    verify_directionality, verify_terms, run_html_verification
)

def post_process_ollama(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Post-process Ollama translations to extract dictionaries and select best translations.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing translated terms.

	Returns
	-------
	pd.DataFrame
		The updated DataFrame with extracted translations.
	"""
	tqdm.pandas(desc="Extracting dictionaries")
	df['ollama_extracted_dictionaries'] = df['ollama_content'].progress_apply(extract_dictionaries_from_string)

	tqdm.pandas(desc="Selecting translated terms")
	df = df.progress_apply(extract_ollama_translated_term, axis=1)

	# Ensure we don't keep an empty value when a translation exists
	df.loc[
		(df.ollama_translated_term.isna()) & (df.ollama_translated_term != df.ollama_translation), 
		'ollama_translated_term'
	] = df['ollama_translation']

	return df

def mark_errored_terms(df: pd.DataFrame, error_file: str, service: str) -> pd.DataFrame:
	"""
	Mark terms and languages that have previously encountered errors when translating in the DataFrame.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing all terms and languages.
	error_file : str
		A path to the error file for logging errors.
	service : str
		A string indicating the translation service (e.g., 'gt', 'enmt', 'openai', 'ollama').
	
	Returns
	--------
	pd.DataFrame
		A DataFrame with errored terms marked.
	"""
	error_col = f'exclude_{service.replace(" ", "_").lower()}'
	if os.path.exists(error_file):
		error_df = read_csv_file(error_file)
		# Normalize term_source column — entries may be stored as a plain string
		# (e.g. "Digital Humanities") or as a Python list-string (e.g. "['Digital Humanities']").
		# Previously only the list-string case was handled, so plain strings were silently
		# dropped, meaning no errors were ever actually excluded.
		expanded_rows = []
		for _, row in error_df.iterrows():
			raw = row['term_source']
			if not isinstance(raw, str):
				continue
			# Try to parse as a Python list first; fall back to treating as a plain string
			try:
				parsed = ast.literal_eval(raw)
				terms = parsed if isinstance(parsed, list) else [parsed]
			except (ValueError, SyntaxError):
				terms = [raw]
			for term in terms:
				expanded_rows.append({
					'language_code': row.get('language_code'),
					'term_source': term
				})

		# Create a DataFrame of (language_code, language_name, term_source) triplets to exclude
		if expanded_rows:
			expanded_error_df = pd.DataFrame(expanded_rows)
			exclude_set = set(expanded_error_df.itertuples(index=False, name=None))

			df[error_col] = df.apply(
				lambda row: (row.get('language_code'), row.get('term_source')) in exclude_set,
				axis=1
			)
		else:
			df[error_col] = False
	else:
		df[error_col] = False

	return df

def read_existing_translations_for_service(
	target_terms: List[str],
	data_directory_path: str,
	translation_column: str,
	subfolder: str = "historic_translations_data",
	file_suffix: str = "_translated_terms.csv",
	timestamp_column: str = "translation_timestamp",
) -> pd.DataFrame:
	"""
	Reads all existing translation files for a specific service and returns the most recent entries per language and term_source.

	Parameters
	----------
	target_terms : List[str]
		A list of terms whose translations to load.
	data_directory_path : str
		Base path to metadata_files/translated_terms.
	translation_column : str
		The column name where the translations are stored (e.g., 'openai_translated_term').
	subfolder : str, optional
		The subdirectory under each term's folder where the service translation files live.
	file_suffix : str, optional
		The suffix used in the translation file name.
	timestamp_column : str, optional
		The column name used for keeping the most recent translation.

	Returns
	-------
	pd.DataFrame
		Combined and deduplicated DataFrame of the most recent translations for the service.
	"""
	dfs = []

	for term_source in target_terms:
		term_slug = term_source.lower().replace(" ", "_")
		service_file = os.path.join(
			data_directory_path,
			"metadata_files",
			"translated_terms",
			term_slug,
			subfolder,
			f"{translation_column}{file_suffix}"
		)

		if os.path.exists(service_file):
			df = read_csv_file(service_file)
			df["term_source"] = term_source
			if timestamp_column in df.columns:
				df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
				df = df.sort_values(timestamp_column, ascending=True).drop_duplicates(
					subset=["language_code", "language_name", "term_source"], keep="last"
				)
			dfs.append(df)

	if dfs:
		return pd.concat(dfs).reset_index(drop=True)
	else:
		return pd.DataFrame()

def get_translation_column(translation_columns: list, required_suffix: str = '_translated_term') -> Optional[str]:
	"""
	Returns the first translation column matching the required suffix, or None if not found.

	Parameters
	----------
	translation_columns : list
		A list of translation column names.
	required_suffix : str, optional
		The suffix to look for in the translation columns (default is '_translated_term').

	Returns
	-------
	str or None
		The first translation column that matches the required suffix, or None if no match is found.
	"""
	for col in translation_columns:
		if required_suffix in col:
			return col
	return None

def load_existing_translations_by_term_sources(
	base_path: str,
	translate_file_name: str,
	term_sources: List[str],
	translation_columns: List[str],
	timestamp_column: str = "translation_timestamp",
	use_cached_translations: bool = True
) -> pd.DataFrame:
	"""
	Loads existing translations from CSV files for specified term sources.
	Parameters
	----------
	base_path : str
		Base path where the term source directories are located.
	translate_file_name : str
		Name of the translation file to read (e.g., 'translated_terms.csv').
	term_sources : List[str]
		List of term sources to load translations for.
	translation_columns : List[str]
		List of translation columns to include in the final DataFrame.
	timestamp_column : str, optional
		Name of the timestamp column to use for deduplication (default is 'translation_timestamp').
	use_cached_translations : bool, optional
		Whether to use cached translations (default is True).

	Returns
	-------
	pd.DataFrame
		A DataFrame containing the most recent translations for each term source, deduplicated by language and term source.
	"""
	all_dfs = []
	for term in term_sources:
		file_path = os.path.join(base_path, term.lower().replace(" ", "_"), "historic_translations_data", translate_file_name)
		if os.path.exists(file_path) and use_cached_translations:
			df = pd.read_csv(file_path)
			df["term_source"] = term
			if timestamp_column in df.columns:
				df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
				df = df.sort_values(timestamp_column, ascending=False).drop_duplicates(
					subset=["language_code", "language_name", "term_source"], keep="first"
				)
				console.print(f"Loaded {len(df)} translations for {term} from {file_path}", style="bold green")
			all_dfs.append(df)

	return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=["language_code", "language_name", "term_source"] + translation_columns)

def process_individual_terms(
	translate_file_path: str,
	translate_file_name: str, 
	final_df: pd.DataFrame, 
	translation_columns: List[str],
	translation_function: Callable[[pd.Series], pd.Series], 
	service_name: str,
	should_use_cached_translations: bool,
	should_override_wikipedia: bool,
	post_process_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
	exclude_errors_file: Optional[str] = None
) -> pd.DataFrame:
	console.print(f"Processing terms for {service_name}", style="bold green")
	console.print(f"Our dataframe has {len(final_df)} rows", style="bold green")

	translation_column = get_translation_column(translation_columns)
	if not translation_column:
		console.print("No translation column found in translation_columns", style="bold red")
		return pd.DataFrame()

	exclude_col = f'exclude_{service_name.lower().replace(" ", "_")}'

	# Exclude Wikipedia translations unless override is set
	if 'wikipedia_translated_term' in final_df.columns and not should_override_wikipedia:
		final_df = final_df[final_df.wikipedia_translated_term.isna()]

	# Mark errored terms (adds exclude_{service} flag)
	if exclude_errors_file:
		final_df = mark_errored_terms(final_df, exclude_errors_file, service=service_name.lower())
		if exclude_col in final_df.columns:
			console.print(f"Excluded terms: {len(final_df[final_df[exclude_col]])}", style="bold red")
			console.print(f"Working terms: {len(final_df[~final_df[exclude_col]])}", style="bold green")
			excluded_terms_df = final_df[final_df[exclude_col]].copy()
	else:
		excluded_terms_df = pd.DataFrame()

	# Load existing translations per term
	existing_translations = load_existing_translations_by_term_sources(
		base_path=translate_file_path,
		translate_file_name=translate_file_name,
		term_sources=final_df.term_source.unique().tolist(),
		translation_columns=translation_columns,
		use_cached_translations=should_use_cached_translations
	)

	if len(existing_translations) > 0:
		console.print(f"Loaded {len(existing_translations)} existing translations", style="bold green")
		merge_cols = ['language_code', 'language_name', 'term_source']
		cols_to_merge = list(set(merge_cols + translation_columns) & set(existing_translations.columns))
		final_df = final_df.merge(existing_translations[cols_to_merge], on=merge_cols, how='left')

	# Identify missing terms
	if translation_column in final_df.columns:
		missing_terms_df = final_df[final_df[translation_column].isna()]
	else:
		missing_terms_df = final_df

	if exclude_col in missing_terms_df.columns:
		missing_terms_df = missing_terms_df[~missing_terms_df[exclude_col]]

	# Translate missing terms
	if not missing_terms_df.empty:
		tqdm.pandas(desc=f"Translating missing terms using {service_name}")
		try:
			# get_ollama_translation returns (row, consecutive_timeouts) to thread the
			# timeout counter through calls without relying on a global. We detect the
			# tuple return here and unwrap it; other translation functions return a plain
			# Series and are called with only the row argument as before.
			consecutive_timeouts = 0
			_fn_accepts_ct = len(inspect.signature(translation_function).parameters) > 1
			def _apply_translation(row):
				nonlocal consecutive_timeouts
				if _fn_accepts_ct:
					result = translation_function(row, consecutive_timeouts)
				else:
					result = translation_function(row)
				if isinstance(result, tuple):
					row_out, consecutive_timeouts = result
					return row_out
				return result
			missing_terms_df = missing_terms_df.progress_apply(_apply_translation, axis=1)
		except Exception as e:
			error_str = str(e)
			if "Ollama service unresponsive" in error_str or "OpenAI quota exceeded" in error_str or "Claude model not found" in error_str:
				console.print(f"\n✓ Saving checkpoint before stopping: {len(missing_terms_df)} rows processed so far", style="bold yellow")
				# Save partial results so the next run can load them as cached translations
				single_path = os.path.join(translate_file_path, "checkpoint_" + translate_file_name)
				missing_terms_df.to_csv(single_path, index=False)
				console.print(f"✓ Checkpoint saved to: {single_path}", style="bold green")
				console.print(f"\nTo resume, run the script again - it will load this checkpoint and continue from where it left off.", style="bold cyan")
				raise
			else:
				raise

		if post_process_function:
			missing_terms_df = post_process_function(missing_terms_df)

		# Filter excluded terms not already in existing translations
		if not excluded_terms_df.empty and {'language_code', 'language_name', 'term_source'}.issubset(excluded_terms_df.columns):
			excluded_terms_df = excluded_terms_df[
				~excluded_terms_df.set_index(['language_code', 'language_name', 'term_source']).index.isin(
					existing_translations.set_index(['language_code', 'language_name', 'term_source']).index
				)
			]
		try:
			def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
				"""Drop any duplicate columns, keeping the first occurrence."""
				return df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

			finalized_translations = pd.concat(
				[
					_dedup_cols(existing_translations),
					_dedup_cols(missing_terms_df),
					_dedup_cols(excluded_terms_df),
				],
				ignore_index=True
			)
		except Exception:
			existing_translations.to_csv("existing_translations.csv", index=False)
			missing_terms_df.to_csv("missing_terms.csv", index=False)
			excluded_terms_df.to_csv("excluded_terms.csv", index=False)
			raise ValueError("Error concatenating DataFrames. Check the saved CSV files for details.")

		console.print(f"Finalized translations have {len(finalized_translations)} rows", style="bold green")

		finalized_translations = finalized_translations.drop_duplicates(
			subset=['language_code', 'language_name', 'term_source', translation_column]
		)

		console.print(f"After dropping duplicates: {len(finalized_translations)} rows", style="bold green")

		# Save each term's translations separately
		for term_source in finalized_translations.term_source.unique():
			single_df = finalized_translations[finalized_translations.term_source == term_source]
			single_path = os.path.join(
				translate_file_path,
				term_source.lower().replace(" ", "_"),
				"historic_translations_data",
				translate_file_name
			)
			os.makedirs(os.path.dirname(single_path), exist_ok=True)
			single_df.to_csv(single_path, index=False)

	else:
		finalized_translations = existing_translations

	# Drop exclude column
	if exclude_col in finalized_translations.columns:
		finalized_translations = finalized_translations.drop(columns=[exclude_col])

	return finalized_translations

def process_grouped_terms(
	translate_file_path: str,
	translate_file_name: str,
	translation_columns: List[str], 
	final_df: pd.DataFrame, 
	translation_function: Callable[[pd.Series], pd.Series], 
	service_name: str,
	should_use_cached_translations: bool,
	should_override_wikipedia: bool,
	exclude_errors_file: Optional[str] = None
) -> pd.DataFrame:
	"""
	Process terms using a specified translation service, only translating terms missing translations for any language.
	Saves the final translations persistently to avoid redundant API calls.

	Parameters
	----------
	translate_file_path : str
		The path to the file storing translations.
	translate_file_name : str
		The name of the file where translations are stored (e.g., 'translated_terms.csv').
	translation_columns : List[str]
		A list of translation column names to include in the final DataFrame (e.g., ['gt_translated_term', 'enmt_translated_term']).
	final_df : pd.DataFrame
		The full DataFrame containing all language codes.
	translation_column : str
		The column name where translations are stored (e.g., 'gt_translated_term', 'enmt_translated_term').
	translation_function : Callable[[pd.Series], pd.Series]
		The function that performs translations for each row.
	service_name : str
		A friendly name for the translation service (for logging purposes).
	should_use_cached_translations : bool
		A boolean indicating whether to use cached translations. If True, files will not load and all data will be rerun.
	should_override_wikipedia: bool
		A boolean indicating whether to override Wikipedia translations.
	exclude_errors_file : Optional[str]
		A path to the error file for logging errors.

	Returns
	-------
	pd.DataFrame
		A DataFrame with all translated terms.
	"""
	console.print(f"Processing terms for {service_name}", style="bold green")
	console.print(f"Our dataframe has {len(final_df)} rows", style="bold green")

	translation_column = get_translation_column(translation_columns)
	if not translation_column:
		console.print("No translation column found in translation_columns", style="bold red")
		return pd.DataFrame()
	# Exclude Wikipedia translations unless override is set
	if "wikipedia_translated_term" in final_df.columns and not should_override_wikipedia:
		final_df = final_df[final_df.wikipedia_translated_term.isna()]

	# Mark error rows
	exclude_col = f'exclude_{service_name.lower().replace(" ", "_")}'
	console.print(f"Excluding terms with {exclude_col} column with {exclude_errors_file}", style="bold red")
	if exclude_errors_file:
		final_df = mark_errored_terms(final_df, exclude_errors_file, service=service_name.lower())

	if exclude_col in final_df.columns:
		excluded_terms_df = final_df[final_df[exclude_col]].copy()
		console.print(f"Excluded terms: {len(excluded_terms_df)}", style="bold red")
		working_df = final_df[~final_df[exclude_col]].copy()
		console.print(f"Working terms: {len(working_df)}", style="bold green")
	else:
		excluded_terms_df = pd.DataFrame()
		working_df = final_df.copy()
	# Load existing translations per term
	existing_translations = load_existing_translations_by_term_sources(
		base_path=translate_file_path,
		translate_file_name=translate_file_name,
		term_sources=working_df.term_source.unique().tolist(),
		translation_columns=translation_columns,
		use_cached_translations=should_use_cached_translations
	)

	if len(existing_translations) > 0:
		console.print(f"Loaded {len(existing_translations)} existing translations", style="bold green")
		merge_cols = ['language_code', 'language_name', 'term_source']
		cols_to_merge = list(set(merge_cols + translation_columns) & set(existing_translations.columns))
		working_df = working_df.merge(existing_translations[cols_to_merge], on=merge_cols, how='left')

	# Identify missing translations
	subset_df = working_df[working_df.get(translation_column).isna()] if translation_column in working_df.columns else working_df
	subset_df = subset_df.reset_index(drop=True)
	# Group for translation
	grouped_terms = subset_df.groupby(
		['language_code', 'language_name']
	).agg({'term_source': lambda x: list(x)}).reset_index()

	if not grouped_terms.empty:
		tqdm.pandas(desc=f"Translating missing terms using {service_name}")
		grouped_terms = grouped_terms.progress_apply(translation_function, axis=1)

		grouped_terms[translation_column] = grouped_terms[translation_column].apply(
			lambda x: [t[1] for t in x] if isinstance(x, list) else x
		)

		# Explode into one row per term
		exploded_grouped_terms = grouped_terms.explode(['term_source', translation_column])

		# Filter excluded terms not already in existing translations
		if not excluded_terms_df.empty and {'language_code', 'language_name', 'term_source'}.issubset(excluded_terms_df.columns):
			excluded_terms_df = excluded_terms_df[
				~excluded_terms_df.set_index(['language_code', 'language_name', 'term_source']).index.isin(
					existing_translations.set_index(['language_code', 'language_name', 'term_source']).index
				)
			]

		def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
			"""Drop any duplicate columns, keeping the first occurrence."""
			return df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

		finalized_translations = pd.concat(
			[_dedup_cols(existing_translations), _dedup_cols(exploded_grouped_terms), _dedup_cols(excluded_terms_df)],
			ignore_index=True
		)
		console.print(f"Finalized translations have {len(finalized_translations)} rows", style="bold green")
		finalized_translations = finalized_translations.drop_duplicates(
			subset=['language_code', 'language_name', 'term_source', translation_column]
		)

		console.print(f"Finalized translations after dropping duplicates have {len(finalized_translations)} rows", style="bold green")
		# Save each term's translations separately
		for term_source in finalized_translations.term_source.unique():
			single_df = finalized_translations[finalized_translations.term_source == term_source]
			single_path = os.path.join(
				translate_file_path,
				term_source.lower().replace(" ", "_"),
				"historic_translations_data",
				translate_file_name
			)
			os.makedirs(os.path.dirname(single_path), exist_ok=True)
			single_df.to_csv(single_path, index=False)
	else:
		finalized_translations = existing_translations

	# Drop the temporary column if it exists
	if exclude_col in finalized_translations.columns:
		finalized_translations = finalized_translations.drop(columns=[exclude_col])

	return finalized_translations


def _merge_translations(base_df: pd.DataFrame, new_df: pd.DataFrame, join_cols: list, how: str = 'outer') -> pd.DataFrame:
	"""
	Merge new translation columns onto base_df without producing _x/_y duplicate suffixes.

	Any column on new_df that already exists on base_df (other than the join keys) is
	dropped from new_df before merging, so the existing values on base_df are preserved
	and pandas never needs to disambiguate with suffixes.
	"""
	overlap = [c for c in new_df.columns if c in base_df.columns and c not in join_cols]
	return base_df.merge(new_df.drop(columns=overlap), on=join_cols, how=how)

def generate_initial_terms(target_terms: list, data_directory_path: str, process_dh: bool, use_gt_translate: bool, use_enmt_translate: bool, use_openai_translate: bool, use_claude_translate: bool, use_ollama_translate: bool,  use_wikipedia: bool, override_wikipedia: bool, rerun_llama: bool, use_cached_translations: bool, return_all_data: bool = False, exclude_previous_errors: bool = False, use_gemini_translate: bool = False, use_lingvanex_translate: bool = False, defer_verification: bool = False) -> pd.DataFrame:
	"""
	Generate a dataframe with translated terms. This function assumes you want to at the very least translate Digital Humanities.

	Parameters
	----------
	target_terms : list
		A list of terms to translate in all ISO 639-1 languages.
	data_directory_path : str
		A path to the directory with the datasets.
	process_dh : bool
		A boolean indicating whether Digital Humanities terms already exist.
	use_gt_translate : bool
		A boolean indicating whether to use Google Cloud Translate.
	use_enmt_translate : bool
		A boolean indicating whether to use EasyNMT.
	use_openai_translate : bool
		A boolean indicating whether to use OpenAI API.
	use_wikipedia : bool
		A boolean indicating whether to check for Wikipedia pages.
	override_wikipedia : bool
		A boolean indicating whether to override existing Wikipedia translations for terms.
	rerun_llama : bool
		A boolean indicating whether to rerun the Ollama model after the initial run. Intended to see if it improves translations with OpenAI data.
	use_cached_translations: bool
		A boolean indicating whether to use cached existing translations. If False, existing files will be ignored and all data will be reprocessed.
	return_all_data: bool
		A boolean indicating whether to return the full dataset or subset to only to existing terms. Defaults to only existing terms.
	exclude_previous_errors: bool
		A boolean indicating whether to exclude previous errors from the dataset. Defaults to False.

	Returns
	-------
	pd.DataFrame
		A dataframe with translated terms.
	"""
	error_dir = os.path.join(data_directory_path, "error_logs", "translation_files")
	os.makedirs(error_dir, exist_ok=True)

	gt_error_file = os.path.join(error_dir, "gt_translation_errors.csv")
	enmt_error_file = os.path.join(error_dir, "enmt_translation_errors.csv")
	openai_error_file = os.path.join(error_dir, "openai_translation_errors.csv")
	claude_error_file = os.path.join(error_dir, "claude_translation_errors.csv")
	first_ollama_error_file = os.path.join(error_dir, "first_ollama_translation_errors.csv")
	ollama_error_file = os.path.join(error_dir, "ollama_translation_errors.csv")
	gemini_error_file = os.path.join(error_dir, "gemini_translation_errors.csv")
	lingvanex_error_file = os.path.join(error_dir, "lingvanex_translation_errors.csv")
	for error_file in [gt_error_file, enmt_error_file, openai_error_file, claude_error_file, first_ollama_error_file, ollama_error_file, gemini_error_file, lingvanex_error_file]:
		clean_write_error_file(error_file, drop_fields=["term_source", "language_code", "error_url"])
	# Make temp dir wherever running the code
	if not os.path.exists("temp"):
		os.makedirs("temp", exist_ok=True)
	# Load ISO 639-1 languages
	iso_languages_path = os.path.join(data_directory_path, "metadata_files", "iso_639_choices.csv")
	iso_languages = read_csv_file(iso_languages_path)
	iso_languages = iso_languages.rename(columns={'name': 'language_name', 'language': 'language_code'})

	# Process Digital Humanities terms if required
	if process_dh:
		dh_df_path = os.path.join(data_directory_path, "metadata_files", "en.Digital humanities.json")
		dh_df = pd.DataFrame([json.load(codecs.open(dh_df_path, 'r', 'utf-8-sig'))])
		dh_df = dh_df.melt()
		dh_df.columns = ['language_code', 'term']
		merged_dh = pd.merge(dh_df, iso_languages, on='language_code', how='outer')
		merged_dh['term_source'] = 'Digital Humanities'

	# Generate a dataframe with all the terms to translate in all ISO 639-1 languages
	languages_dfs = []
	for term in target_terms:
		term_df = iso_languages.copy()
		term_df['term_source'] = term
		languages_dfs.append(term_df)

	if process_dh:
		languages_dfs.append(merged_dh)

	final_df = pd.concat(languages_dfs).reset_index(drop=True)
	console.print(f"Number of languages to translate terms into: {len(final_df)}", style="bright_cyan")

	# Check if Wikipedia pages exist for the terms
	if use_wikipedia:
		# Check if Wikipedia pages exist for the terms
		unique_term_sources = final_df['term_source'].unique()
		translations_dfs = []
		# Create a file to store errors
		wikipedia_error_file = os.path.join(error_dir, "wikipedia_translation_errors.csv")
		clean_write_error_file(wikipedia_error_file, drop_fields=["term_source", "language_code", "error_url"])
		# Load any previous errors
		if os.path.exists(wikipedia_error_file):
			existing_errors_df = read_csv_file(wikipedia_error_file)
			errored_terms = set(existing_errors_df['term_source'].dropna().unique())
		else:
			errored_terms = set()
		for term_source in unique_term_sources:
			if term_source in errored_terms:
				console.print(f"Skipping {term_source} due to previous errors", style="bold red")
				continue
			folder_term_source = term_source.replace(" ", "_").lower()
			translation_file_path = os.path.join(data_directory_path, "metadata_files", "translated_terms", folder_term_source, "historic_translations_data", "wikipedia_translations.csv")
			if os.path.exists(translation_file_path) and use_cached_translations:
				translations_df = read_csv_file(translation_file_path)
				console.print(f"Loaded Wikipedia translations for {term_source} from file", style="bold green")
				if 'translation_timestamp' in translations_df.columns:
					translations_df['translation_timestamp'] = pd.to_datetime(translations_df['translation_timestamp'])
					# Keep only the most recent translations
					translations_df = translations_df.sort_values(by='translation_timestamp').drop_duplicates(subset=['language_code'], keep='last')
				translations_dfs.append(translations_df)
			else:
				translations = check_if_wikipedia_page_exists(term_source, error_file_path=wikipedia_error_file, console=console, log_error_to_file=log_error_to_file)
				if translations:
					translations_df = pd.DataFrame([translations])
					translations_df = translations_df.T.reset_index().rename(columns={"index": "language_code", 0: "wikipedia_translated_term"})
					translations_df['term_source'] = term_source
					translations_df['translation_timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
					# write translations to a file
					translations_df.to_csv(translation_file_path, index=False)
					console.print(f"Added Wikipedia translations for {term_source}", style="bold green")
					translations_dfs.append(translations_df)
		if translations_dfs:
			final_translations_df = pd.concat(translations_dfs).reset_index(drop=True)
			if len(final_translations_df) > 0:
				merge_cols = list(set(final_df.columns.tolist()) & set(final_translations_df.columns.tolist()))
				final_df = final_df.merge(final_translations_df, on=merge_cols, how='outer')
				console.print(f"Added Wikipedia translations for {term_source}", style="bold green")
				console.print(f"Number of languages with Wikipedia translations: {len(final_df[final_df.wikipedia_translated_term.notna()])}", style="bright_magenta")
				console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")
			else:
				console.print("No Wikipedia translations found", style="bold red")
		else:
			console.print("No Wikipedia translations found", style="bold red")
				
	# Translate the terms using Google Cloud Translate
	terms_file_path = os.path.join(data_directory_path, "metadata_files", "translated_terms")
	if use_gt_translate:
		exploded_grouped_gt_terms = process_grouped_terms(
			translate_file_path=terms_file_path,
			translate_file_name="gt_translations.csv",
			translation_columns=['gt_translated_term'],
			final_df=final_df,
			translation_function=lambda row: get_gt_translation(row, gt_error_file, console, translate_client),
			service_name="Google Cloud Translate",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=gt_error_file if exclude_previous_errors else None
		)

		final_df = _merge_translations(final_df, exploded_grouped_gt_terms, ['language_code', 'language_name', 'term_source'])

		console.print(
			f"Number of languages with Google Translate translations: {len(final_df[final_df.gt_translated_term.notna()])}", 
			style="bright_magenta"
		)
		console.print(
			f"Size of dataset currently {len(final_df)}", 
			style="bright_cyan"
		)

	# Translate the terms using EasyNMT
	if use_enmt_translate:
		exploded_grouped_enmt_terms = process_grouped_terms(
			translate_file_path=terms_file_path,
			translate_file_name="enmt_translations.csv",
			translation_columns=['enmt_translated_term'],
			final_df=final_df,
			translation_function=lambda row: get_enmt_translation(row, enmt_error_file, console, translate_client, model, lambda target_lang: is_enmt_model_available(target_lang, console)),
			service_name="EasyNMT",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=enmt_error_file if exclude_previous_errors else None
		)

		final_df = _merge_translations(final_df, exploded_grouped_enmt_terms, ['language_code', 'language_name', 'term_source'])

		console.print(
			f"Number of languages with EasyNMT translations: {len(final_df[final_df.enmt_translated_term.notna()])}", 
			style="bright_magenta"
		)
		console.print(
			f"Size of dataset currently {len(final_df)}", 
			style="bright_cyan"
		)

	# Translate the terms using Lingvanex (free, top-ranked primary service per Kraus et al. 2025)
	if use_lingvanex_translate:
		exploded_grouped_lingvanex_terms = process_grouped_terms(
			translate_file_path=terms_file_path,
			translate_file_name="lingvanex_translations.csv",
			translation_columns=['lingvanex_translated_term'],
			final_df=final_df,
			translation_function=lambda row: get_lingvanex_translation(row, lingvanex_error_file, console),
			service_name="Lingvanex",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=lingvanex_error_file if exclude_previous_errors else None
		)
		final_df = _merge_translations(final_df, exploded_grouped_lingvanex_terms, ['language_code', 'language_name', 'term_source'])
		console.print(
			f"Number of languages with Lingvanex translations: {len(final_df[final_df.lingvanex_translated_term.notna()])}",
			style="bright_magenta"
		)
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate the terms using Ollama
	if use_ollama_translate:

		final_ollama_translations = process_individual_terms(
			translate_file_path=terms_file_path,
			translate_file_name="ollama_translations.csv",
			translation_columns=['ollama_translated_term', 'ollama_content', 'ollama_created_at', 'ollama_done_reason',
			'ollama_eval_count', 'ollama_eval_duration',
			'ollama_extracted_dictionaries', 'ollama_load_duration', 'ollama_model',
			'ollama_prompt_eval_count', 'ollama_prompt_eval_duration',
			'ollama_total_duration', 'ollama_translation'],
			final_df=final_df,
			translation_function=lambda row, ct=0: get_ollama_translation(row, first_ollama_error_file, console, current_prompt_variant, current_term_contexts, get_prompt, log_error_to_file, ollama_model='llama3.1', request_delay=2.0, request_timeout=120, consecutive_timeouts=ct),
			service_name="Ollama",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			post_process_function=post_process_ollama,
			exclude_errors_file=first_ollama_error_file if exclude_previous_errors else None
		)

		# Extract all columns that include 'ollama'
		ollama_cols = [col for col in final_ollama_translations.columns if 'ollama' in col]

		# Merge results back
		final_df = _merge_translations(final_df, final_ollama_translations[['language_code', 'language_name', 'term_source'] + ollama_cols], ['language_code', 'language_name', 'term_source'])
		console.print(f"Number of languages with Ollama translations: {len(final_df[final_df.ollama_translated_term.notna()])}", style="bright_magenta")
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate the terms using OpenAI
	if use_openai_translate:
		final_openai_translations = process_individual_terms(
			translate_file_path=terms_file_path,
			translate_file_name="openai_translations.csv",
			translation_columns=['openai_translated_term', 'openai_completion_tokens', 'openai_created', 'openai_finish_reason',
			'openai_model', 'openai_prompt_tokens', 'openai_total_tokens',
			'openai_translation_rationale', 'openai_translation'],
			final_df=final_df,
			translation_function=lambda row: get_openai_translation(row, openai_error_file, console, client, current_prompt_variant, current_term_contexts, get_prompt, parse_translation_response, log_error_to_file),
			service_name="OpenAI",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=openai_error_file if exclude_previous_errors else None
		)

		# Extract all columns that include 'openai'
		openai_cols  = [col for col in final_openai_translations.columns.tolist() if 'openai' in col]
		# Merge results back
		final_df = _merge_translations(final_df, final_openai_translations[['language_code', 'language_name', 'term_source'] + openai_cols], ['language_code', 'language_name', 'term_source'])
		console.print(f"Number of languages with OpenAI translations: {len(final_df[final_df.openai_translated_term.notna()])}", style="bright_magenta")
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate the terms using Claude
	if use_claude_translate:
		final_claude_translations = process_individual_terms(
			translate_file_path=terms_file_path,
			translate_file_name="claude_translations.csv",
			translation_columns=['claude_translated_term', 'claude_output_tokens', 'claude_created', 'claude_stop_reason',
			'claude_model', 'claude_input_tokens', 'claude_total_tokens',
			'claude_translation_rationale', 'claude_translation'],
			final_df=final_df,
			translation_function=lambda row: get_claude_translation(row, claude_error_file, console, claude_client, current_prompt_variant, current_term_contexts, get_prompt, parse_translation_response, log_error_to_file),
			service_name="Claude",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=claude_error_file if exclude_previous_errors else None
		)

		# Extract all columns that include 'claude'
		claude_cols = [col for col in final_claude_translations.columns.tolist() if 'claude' in col]
		# Merge results back
		final_df = _merge_translations(final_df, final_claude_translations[['language_code', 'language_name', 'term_source'] + claude_cols], ['language_code', 'language_name', 'term_source'])
		console.print(f"Number of languages with Claude translations: {len(final_df[final_df.claude_translated_term.notna()])}", style="bright_magenta")
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate the terms using Gemini (gemini-2.0-flash, top-ranked LLM per Kraus et al. 2025)
	if use_gemini_translate:
		final_gemini_translations = process_individual_terms(
			translate_file_path=terms_file_path,
			translate_file_name="gemini_translations.csv",
			translation_columns=['gemini_translated_term', 'gemini_translation_rationale', 'gemini_model', 'gemini_created'],
			final_df=final_df,
			translation_function=lambda row: get_gemini_translation(row, gemini_error_file, console, gemini_client, GEMINI_MODEL, current_prompt_variant, current_term_contexts, get_prompt, parse_translation_response, log_error_to_file),
			service_name="Gemini",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=gemini_error_file if exclude_previous_errors else None
		)
		gemini_cols = [col for col in final_gemini_translations.columns.tolist() if 'gemini' in col]
		final_df = _merge_translations(final_df, final_gemini_translations[['language_code', 'language_name', 'term_source'] + gemini_cols], ['language_code', 'language_name', 'term_source'])
		console.print(f"Number of languages with Gemini translations: {len(final_df[final_df.gemini_translated_term.notna()])}", style="bright_magenta")
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Retranslate Ollama terms if rerun_llama is True (only for comparative variant)
	if use_ollama_translate and rerun_llama and current_prompt_variant == 'comparative':
		# Rename previous Ollama columns to store first-run results
		if any(col.startswith('ollama') for col in final_df.columns):
			rename_mapping = {col: col.replace('ollama', 'first_ollama') for col in final_df.columns if col.startswith('ollama')}
			final_df = final_df.rename(columns=rename_mapping)

		final_second_ollama_translations = process_individual_terms(
			translate_file_path=terms_file_path,
			translate_file_name="second_ollama_translations.csv",
			translation_columns=['ollama_translated_term', 'ollama_content', 'ollama_created_at', 'ollama_done_reason',
			'ollama_eval_count', 'ollama_eval_duration',
			'ollama_extracted_dictionaries', 'ollama_load_duration', 'ollama_model',
			'ollama_prompt_eval_count', 'ollama_prompt_eval_duration',
			'ollama_total_duration', 'ollama_translation'],
			final_df=final_df,
			translation_function=lambda row, ct=0: get_ollama_translation(row, ollama_error_file, console, current_prompt_variant, current_term_contexts, get_prompt, log_error_to_file, ollama_model='llama3.1', request_delay=2.0, request_timeout=120, consecutive_timeouts=ct),
			service_name="Ollama",  # Same service name as first pass to share the exclude column
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			post_process_function=post_process_ollama,
			exclude_errors_file=ollama_error_file if exclude_previous_errors else None
		)

		# Extract all new Ollama-related columns
		second_ollama_cols = [col for col in final_second_ollama_translations.columns if col.startswith('ollama')]

		# Merge second-pass Ollama translations into final_df
		final_df = _merge_translations(final_df, final_second_ollama_translations[['language_code', 'language_name', 'term_source'] + second_ollama_cols], ['language_code', 'language_name', 'term_source'], how='left')

		console.print(f"Number of languages with second-pass Ollama translations: {len(final_df[final_df.ollama_translated_term.notna()])}", style="bright_magenta")
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	console.print(f"Columns of final_df currently after getting all terms: {final_df.columns}", style="bright_cyan")

	# Ask user if they want to continue
	check_if_continue = console.input("Do you want to continue? (y/n): ")
	if check_if_continue == 'n':
		sys.exit()
	
	# Clean up the final DataFrame
	cleaned_df = final_df.reset_index(drop=True)

	# Initialize 'term' column if not present
	if 'term' not in cleaned_df.columns:
		cleaned_df['term'] = None

	# Prioritize translations in order of preference
	translation_sources = ['wikipedia_translated_term', 'openai_translated_term', 'claude_translated_term',
						   'gemini_translated_term', 'gt_translated_term', 'lingvanex_translated_term',
						   'ollama_translated_term', 'first_ollama_translated_term', 'enmt_translated_term']

	for source in translation_sources:
		if source in cleaned_df.columns:
			cleaned_df.loc[(cleaned_df[source].notna()) & (cleaned_df.term.isna()), 'term'] = cleaned_df[source]

	console.print(f"Number of languages with terms: {len(cleaned_df[cleaned_df.term.notna()])}", style="bright_magenta")

	# Ensure we only apply html.unescape to non-None values
	for col in translation_sources + ['term']:
		if col in cleaned_df.columns:
			cleaned_df[col] = cleaned_df[col].apply(lambda x: html.unescape(x) if pd.notna(x) else None)

	# Ask user if they want to delete temp files
	if console.input("Do you want to delete the temp folder? (y/n): ") == 'y':
		try:
			shutil.rmtree("temp")
		except FileNotFoundError:
			console.print("Temp folder already deleted.", style="bold yellow")

	if not return_all_data:
		cleaned_df = cleaned_df[cleaned_df.term.notna()]

	return cleaned_df

def combine_language_data(directionality_df: pd.DataFrame, translated_terms_df: pd.DataFrame, data_directory_path:str, should_skip_existing_files:bool, use_html_verification: bool = True, defer_verification: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
	# Subset directionality to LTR and RTL languages
	directionality_df = directionality_df[directionality_df.directionality.isin(['ltr', 'rtl'])]

	# Merge the directionality data with the translated terms
	merged_lang_terms_df = pd.merge(directionality_df[['language_code', 'directionality', 'English language name', 'local language name', 'comment', 'local or English Wikipedia article']], translated_terms_df, on='language_code', how="outer")
	merged_lang_terms_df = merged_lang_terms_df[merged_lang_terms_df.language_code != "see also Test languages"]

	console.print(f"Our data now contains info for {merged_lang_terms_df[merged_lang_terms_df.term.notna()]['English language name'].nunique()} but we also are missing terms for the following number of languages {merged_lang_terms_df[merged_lang_terms_df.term.isna()]['English language name'].nunique()}", style="bold green")
	merged_lang_terms_df = merged_lang_terms_df[merged_lang_terms_df.term.notna()]
	# Fill NaN values with an empty string
	merged_lang_terms_df['English language name'] = merged_lang_terms_df['English language name'].fillna('').astype(str)
	merged_lang_terms_df['local language name'] = merged_lang_terms_df['local language name'].fillna('').astype(str)
	merged_lang_terms_df['local or English Wikipedia article'] = merged_lang_terms_df['local or English Wikipedia article'].fillna('').astype(str)
	merged_lang_terms_df['comment'] = merged_lang_terms_df['comment'].fillna('').astype(str)
	merged_lang_terms_df = merged_lang_terms_df.reset_index(drop=True)
	processed_translations_dfs = []
	grouped_translations_dfs = []
	for term_source in merged_lang_terms_df.term_source.unique():
		console.print(f"Processing term: {term_source}", style="bold green")
		lower_term = term_source.lower().replace(" ", "_")
		subset_merged_lang_terms_df = merged_lang_terms_df[merged_lang_terms_df.term_source == term_source]
		processed_translated_terms_output_path = get_variant_output_path(os.path.join(data_directory_path, "metadata_files", "translated_terms", lower_term), "processed_translated_terms.csv", current_prompt_variant)
		if os.path.exists(processed_translated_terms_output_path) and not should_skip_existing_files:
			existing_subset_merged_lang_terms_df = read_csv_file(processed_translated_terms_output_path)
			subset_merged_lang_terms_df = pd.concat([subset_merged_lang_terms_df, existing_subset_merged_lang_terms_df])
			subset_merged_lang_terms_df = subset_merged_lang_terms_df.drop_duplicates(subset=['term', 'language_code'])
		subset_merged_lang_terms_df.to_csv(f'{processed_translated_terms_output_path}', index=False)
		subset_merged_lang_terms_df = subset_merged_lang_terms_df.reset_index(drop=True)
		processed_merged_lang_terms_df = verify_terms(console, subset_merged_lang_terms_df, processed_translated_terms_output_path, use_html_verification=use_html_verification, defer_verification=defer_verification)
		processed_translations_dfs.append(processed_merged_lang_terms_df)
		grouped_terms_df = processed_merged_lang_terms_df.groupby(['term_source', 'term']).agg({
			'language_code': ', '.join,
			'term': 'count',
			'directionality': lambda x: ','.join(set(x.dropna())),
			'English language name': lambda x: ', '.join(x.dropna()),
			'local language name': lambda x: ', '.join(set(x.dropna())),
			'local or English Wikipedia article': lambda x: ', '.join(set(x.dropna())),
			'comment': lambda x: ', '.join(set(x.dropna())),
		}).reset_index(level=0)
		grouped_terms_df = grouped_terms_df.rename(columns={'term': 'counts'})
		grouped_terms_df['term'] = grouped_terms_df.index
		grouped_terms_df = grouped_terms_df.reset_index(drop=True)
		grouped_terms_output_path = get_variant_output_path(os.path.join(data_directory_path, "metadata_files", "translated_terms", lower_term), "grouped_translated_terms.csv", current_prompt_variant)
		if os.path.exists(grouped_terms_output_path) and not should_skip_existing_files:
			existing_grouped_terms_df = read_csv_file(grouped_terms_output_path)
			grouped_terms_df = pd.concat([grouped_terms_df, existing_grouped_terms_df])
			grouped_terms_df = grouped_terms_df.drop_duplicates(subset=['term', 'term_source', 'counts'])
		grouped_terms_df = verify_directionality(console, grouped_terms_df, grouped_terms_output_path, directionality_df)
		grouped_translations_dfs.append(grouped_terms_df)
	processed_translations_df = pd.concat(processed_translations_dfs)
	grouped_translations_df = pd.concat(grouped_translations_dfs)
	return processed_translations_df, grouped_translations_df

def save_translated_terms(translated_terms_df: pd.DataFrame, updated_target_terms: list, data_directory_path: str, ) -> None:
	"""
	Save initial translated terms to CSV files for each target term.

	Parameters
	----------
	translated_terms_df : pd.DataFrame
		The DataFrame containing initial translated terms.
	updated_target_terms : list
		A list of target terms to save.
	data_directory_path : str
		The path to the directory where the files will be saved.
	
	Returns
	-------
	None
	"""
	for term in updated_target_terms:
		 
		lower_term = term.lower().replace(" ", "_")
		subset_translated_terms_df = translated_terms_df[translated_terms_df.term_source == term]
		
		translated_terms_output_path = get_variant_output_path(os.path.join(data_directory_path, "metadata_files", "translated_terms", lower_term), "initial_translated_terms.csv", current_prompt_variant)
		subset_translated_terms_df.to_csv(translated_terms_output_path, index=False)

def load_existing_data(file_paths: List[str], skip_existing_files: bool) -> Tuple[bool, List[pd.DataFrame]]:
	"""
	Check and load existing data from a list of file paths.

	Parameters:
		file_paths (List[str]): A list of file paths to check for existing data.
		skip_existing_files (bool): A flag indicating whether to load existing datasets

	Returns:
		(bool, List[pd.DataFrame]): A flag indicating if all data exists and a list of dataframes loaded.
	"""
	all_files_exist = all(os.path.exists(path) for path in file_paths)
	if all_files_exist and not skip_existing_files:
		return True, [read_csv_file(path) for path in file_paths]
	return False, []

def generate_translated_terms(data_directory_path: str, target_terms: List[str], directionality_df: pd.DataFrame, gt_translate: bool, enmt_translate: bool, openai_translate: bool, claude_translate: bool, ollama_translate: bool, wikipedia_translate: bool, override_wikipedia: bool, run_llama_twice: bool, skip_cached_files: bool, cached_translations: bool, return_full_terms_data: bool, excluding_previous_errors: bool, use_html_verification: bool = True, defer_verification: bool = False, prompt_variant: str = 'comparative', gemini_translate: bool = False, lingvanex_translate: bool = False, term_contexts: dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Generate translated terms for a list of target terms.

	Parameters
	----------

	data_directory_path : str
		The path to the directory with the datasets.
	target_terms : list
		A list of target terms to translate in all ISO 639-1 languages.
	directionality_df : pd.DataFrame
		A dataframe with language directionality.
	gt_translate : bool
		A boolean indicating whether to use Google Cloud Translate.
	enmt_translate : bool
		A boolean indicating whether to use EasyNMT.
	openai_translate : bool
		A boolean indicating whether to use OpenAI API.
	wikipedia_translate : bool
		A boolean indicating whether to check for Wikipedia pages.
	run_llama_twice : bool
		A boolean indicating whether to rerun Llama 3.2 after OpenAI translation.
	skip_cached_files : bool
		A boolean indicating whether to skip existing files. If True, it will regenerate all translations.
	cached_translations : bool
		A boolean indicating whether to use existing translations. If True, it will skip generating new translations.
	return_full_terms_data: bool
		A boolean indicating whether to return the full translated terms dataset including all nulls.
	excluding_previous_errors: bool
		A boolean indicating whether to exclude previous errors from the dataset.
	use_html_verification: bool
		A boolean indicating whether to use HTML verification interface instead of CLI. Defaults to True.
	prompt_variant : str
		The prompt variant to use for LLM translations. Options: 'minimal', 'comparative', 'expert_persona', 'contextual', 'native_rationale'. Defaults to 'comparative'.

	Returns
	-------
	Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
		A tuple with the combined translated, processed, and grouped dataframes.
	"""
	global current_prompt_variant, current_term_contexts
	current_prompt_variant = prompt_variant
	current_term_contexts = term_contexts or {}
	console.print(f"Using prompt variant: {prompt_variant}", style="bold cyan")

	combined_translated_dfs, combined_processed_dfs, combined_grouped_dfs = [], [], []
	updated_target_terms = []

	for term in target_terms:
		console.print(f"Processing term: {term}", style="bold green")
		lower_term = term.lower().replace(" ", "_")
		file_paths = [
			os.path.join(data_directory_path, "metadata_files", "translated_terms", lower_term, f"{file}.csv")
			for file in ["initial_translated_terms", "processed_translated_terms", "grouped_translated_terms"]
		]

		exists, loaded_dfs = load_existing_data(file_paths, skip_cached_files)
		if exists:
			console.print(f"Loading existing data for term: {term}", style="bold green")
			combined_translated_dfs.append(loaded_dfs[0])
			combined_processed_dfs.append(loaded_dfs[1])
			combined_grouped_dfs.append(loaded_dfs[2])
		else:
			console.print(f"Missing or outdated data for term: {term}", style="bold blue")
			updated_target_terms.append(term)

	if updated_target_terms or len(combined_translated_dfs) == 0:
		console.print(f"Generating translations for: {updated_target_terms}", style="bold green")
		new_translated_terms_df = generate_initial_terms(
			updated_target_terms, data_directory_path, 'Digital Humanities' in updated_target_terms,
			gt_translate, enmt_translate, openai_translate, claude_translate, ollama_translate,
			wikipedia_translate, override_wikipedia, run_llama_twice, cached_translations, return_full_terms_data, excluding_previous_errors,
			use_gemini_translate=gemini_translate, use_lingvanex_translate=lingvanex_translate,
			defer_verification=defer_verification
		)
		save_translated_terms(new_translated_terms_df, updated_target_terms, data_directory_path)
		combined_translated_dfs.append(new_translated_terms_df)

	# Concatenate results
	combined_translated_df = pd.concat(combined_translated_dfs, ignore_index=True) if combined_translated_dfs else pd.DataFrame()

	if not combined_translated_df.empty:
		# Run HTML verification if enabled
		if use_html_verification:
			combined_translated_df = run_html_verification(console, combined_translated_df, data_directory_path)

		combined_processed_df, combined_grouped_df = combine_language_data(directionality_df, combined_translated_df, data_directory_path, skip_cached_files, use_html_verification=use_html_verification, defer_verification=defer_verification)
	else:
		console.print("No data available for processing. Skipping combine_language_data.", style="bold yellow")
		combined_processed_df, combined_grouped_df = pd.DataFrame(), pd.DataFrame()

	return combined_translated_df, combined_processed_df, combined_grouped_df


if __name__ == '__main__':
	local_data_directory_path = get_data_directory_path()
	existing_directionality_path = os.path.join(local_data_directory_path, "metadata_files", "iso_639_choices_directionality_wikimedia.csv")
	local_target_terms: list = ["Computational Humanities"]

	existing_directionality_df = get_directionality(existing_directionality_path)
	should_use_gt_translate = True # Use Google Cloud Translate
	should_use_enmt_translate = True # Use EasyNMT
	should_use_openai_translate = True # Use OpenAI API
	should_use_claude_translate = True # Use Claude API
	should_use_ollama_translate = True # Use Llama 3.1 (Ollama)
	should_use_wikipedia_translate = True # Check for Wikipedia pages
	should_override_wikipedia = True # Override Wikipedia translations
	should_run_llama_twice = True # Rerun Llama 3.1 after Claude translation
	should_skip_cached_files = False # Skip cached files (False = use cached files)
	should_use_cached_translations = True # Use cached translations
	should_return_all_terms = False # Return all terms
	should_exclude_previous_errors = True # Exclude previous errors
	should_use_html_verification = True # Use HTML verifier instead of CLI (opens in browser)
	should_use_gemini_translate = False # Use Gemini (gemini-2.0-flash) — top-ranked LLM in Kraus et al. 2025
	should_use_lingvanex_translate = False # Use Lingvanex — top-ranked primary service in Kraus et al. 2025

	combined_translated_df, combined_processed_df, combined_grouped_df = generate_translated_terms(local_data_directory_path, local_target_terms, existing_directionality_df, should_use_gt_translate, should_use_enmt_translate, should_use_openai_translate, should_use_claude_translate, should_use_ollama_translate, should_use_wikipedia_translate, should_override_wikipedia, should_run_llama_twice, should_skip_cached_files, should_use_cached_translations, should_return_all_terms, should_exclude_previous_errors, should_use_html_verification, gemini_translate=should_use_gemini_translate, lingvanex_translate=should_use_lingvanex_translate)